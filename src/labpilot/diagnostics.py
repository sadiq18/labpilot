"""Environment diagnostics for the pieces LabPilot depends on but doesn't control.

Failures here (wrong Python version, LightGBM/libomp not loadable, missing
Kaggle credentials) surface as confusing tracebacks deep inside a pipeline run
if left unchecked. `check_environment()` runs the same checks up front so the
CLI can fail fast with a plain-English fix instead.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from labpilot.config import Settings

if TYPE_CHECKING:
    from rich.console import Console

MIN_PYTHON = (3, 11)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def _check_python_version() -> CheckResult:
    version = sys.version_info
    ok = (version.major, version.minor) >= MIN_PYTHON
    detail = f"{version.major}.{version.minor}.{version.micro}"
    fix = f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ (e.g. `uv python install 3.11`)."
    return CheckResult("Python version", ok, detail, "" if ok else fix)


def kaggle_credentials_present() -> bool:
    """True when LabPilot can see any configured Kaggle credential source."""
    settings = Settings()
    has_token = bool(settings.kaggle_api_token)
    has_legacy = bool(settings.kaggle_username) and bool(settings.kaggle_key)
    has_token_file = (Path.home() / ".kaggle" / "access_token").exists()
    has_legacy_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_token or has_legacy or has_token_file or has_legacy_file


def _check_kaggle_credentials() -> CheckResult:
    # Mirrors how `load_config()` actually resolves credentials (env vars or
    # a local .env file via pydantic Settings), not just raw os.environ, so
    # this check can't disagree with what a real run will do.
    ok = kaggle_credentials_present()
    detail = "found" if ok else "not found"
    from labpilot.workspace import discover_workspace

    if ok:
        fix = ""
    else:
        workspace = discover_workspace()
        env_path = (
            workspace.root / ".env"
            if workspace is not None
            else Path.cwd() / ".env"
        )
        fix = (
            f"Create {env_path} with KAGGLE_API_TOKEN "
            "(see docs/research-pipeline/SOP.md § Credentials)."
        )
    return CheckResult("Kaggle credentials", ok, detail, fix)


def _check_lightgbm() -> CheckResult:
    try:
        import lightgbm  # noqa: F401
    except OSError as exc:
        fix = (
            "brew install libomp"
            if sys.platform == "darwin"
            else "Install your platform's OpenMP runtime (e.g. `apt install libgomp1`)."
        )
        return CheckResult("LightGBM import", False, str(exc), fix)
    except ImportError as exc:
        return CheckResult("LightGBM import", False, str(exc), 'Run: pip install -e ".[dev,llm]"')
    return CheckResult("LightGBM import", True, "importable")


def _check_image_deps() -> CheckResult:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            "Image deps (torch/torchvision)",
            False,
            str(exc),
            'Run: pip install -e ".[image]"',
        )
    return CheckResult("Image deps (torch/torchvision)", True, "importable")


def _check_deep_deps() -> CheckResult:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return CheckResult(
            "Deep deps (torch/torchvision/transformers)",
            False,
            str(exc),
            'Run: pip install -e ".[deep]"',
        )
    return CheckResult("Deep deps (torch/torchvision/transformers)", True, "importable")


def _check_llm_provider() -> CheckResult:
    """Verify the *resolved* LLM provider can actually serve the configured model.

    Every intelligence stage (analyzers, codegen, conductor policy) soft-fails to
    template text when the client is unusable, so a broken provider shows up as
    silently degraded output rather than an error. Checking it here makes
    ``llm_unavailable`` diagnosable instead of mysterious.
    """
    name = "LLM provider"
    try:
        from labpilot.config import load_config

        config = load_config().llm
    except Exception as exc:  # noqa: BLE001 — config errors are reported, not raised
        return CheckResult(name, False, f"config load failed: {exc}", "Check configs/default.yaml")

    provider = (config.provider or "").strip().lower()
    model = (config.model or "").strip()

    if provider == "ollama":
        from labpilot.llm.ollama import OllamaProvider

        probe = OllamaProvider(config.ollama_base_url)
        if not probe.is_reachable(timeout_seconds=2.0):
            return CheckResult(
                name,
                False,
                f"ollama unreachable at {config.ollama_base_url}",
                "Start Ollama (`ollama serve`) or fix llm.ollama_base_url.",
            )
        available = probe.list_models()
        # Ollama accepts a bare name for an explicitly ":latest" tag.
        if model and model not in available and f"{model}:latest" not in available:
            return CheckResult(
                name,
                False,
                f"ollama up but model {model!r} not pulled "
                f"(have: {', '.join(available[:3]) or 'none'})",
                f"Run: ollama pull {model}",
            )
        return CheckResult(name, True, f"ollama · {model} · {config.ollama_base_url}")

    from labpilot.config import Settings

    settings = Settings()
    key_by_provider = {
        "gemini": settings.gemini_api_key,
        "openai": settings.openai_api_key,
    }
    if provider in key_by_provider:
        if not key_by_provider[provider]:
            return CheckResult(
                name,
                False,
                f"{provider} selected but no API key found",
                f"Set {provider.upper()}_API_KEY in .env, or use LABPILOT_LLM_PROVIDER=ollama.",
            )
        return CheckResult(name, True, f"{provider} · {model}")

    return CheckResult(
        name,
        False,
        f"unknown provider {provider!r}",
        "Set llm.provider to one of: ollama, gemini, openai.",
    )


def check_llm_roles() -> list[CheckResult]:
    """Report the provider resolved for each role, or why there is none.

    A role with no capable provider is a **failure**, not a warning: the work it
    covers will either not run or run on something that cannot do it, and the
    second is worse because it produces plausible wrong answers.

    Returns an empty list when routing is unconfigured — that workspace is still
    on the legacy provider-priority path and has nothing to report.
    """
    try:
        from labpilot.llm.client import build_gateway
        from labpilot.workspace import load_config_for_cwd

        # Workspace-aware on purpose. `load_config()` alone reads only the
        # package default, where `routing` is deliberately empty — so doctor
        # would report nothing about roles precisely in the workspaces that
        # configure them, which is the only place the answer matters.
        config = load_config_for_cwd()[0].llm
    except Exception as exc:  # noqa: BLE001 — config errors are reported, not raised
        return [CheckResult("LLM routing", False, f"config load failed: {exc}", "")]

    gateway = build_gateway(config)
    if gateway is None:
        return []

    results: list[CheckResult] = []
    for role in sorted(config.routing.roles):
        if role == "default":
            continue
        decision = gateway.preview(role)
        label = f"LLM role · {role}"
        if decision.provider is None:
            results.append(
                CheckResult(
                    label,
                    False,
                    decision.reason,
                    f"Add a provider to llm.routing.providers that satisfies role {role!r}.",
                )
            )
            continue
        detail = f"{decision.provider.name} · {decision.model} ({decision.provider.tier})"
        if decision.degraded:
            detail += " — degraded"
        results.append(CheckResult(label, True, detail))
    return results


def check_environment(include_optional: bool = True) -> list[CheckResult]:
    """Run all environment checks and return their results, in report order."""
    results = [
        _check_python_version(),
        _check_lightgbm(),
        _check_kaggle_credentials(),
        _check_llm_provider(),
        *check_llm_roles(),
    ]
    if include_optional:
        results.extend([_check_image_deps(), _check_deep_deps()])
    return results


OPTIONAL_CHECK_NAMES = frozenset(
    {
        "Image deps (torch/torchvision)",
        "Deep deps (torch/torchvision/transformers)",
    }
)


def required_environment_checks(skip_lightgbm: bool = False) -> list[CheckResult]:
    """Checks that must pass before running pipeline commands."""
    return [
        result
        for result in check_environment(include_optional=False)
        if not (skip_lightgbm and result.name == "LightGBM import")
    ]


def print_diagnostics_report(results: list[CheckResult], console: "Console | None" = None) -> bool:
    """Print a pass/fail table for `results`. Returns True iff everything passed."""
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    table = Table(title="LabPilot Environment Check")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Fix")

    all_ok = True
    for result in results:
        all_ok = all_ok and result.ok
        status = "[green]OK[/green]" if result.ok else "[red]FAIL[/red]"
        table.add_row(result.name, status, result.detail, result.fix)

    console.print(table)
    return all_ok
