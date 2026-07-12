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


def _check_kaggle_credentials() -> CheckResult:
    # Mirrors how `load_config()` actually resolves credentials (env vars or
    # a local .env file via pydantic Settings), not just raw os.environ, so
    # this check can't disagree with what a real run will do.
    settings = Settings()
    has_token = bool(settings.kaggle_api_token)
    has_legacy = bool(settings.kaggle_username) and bool(settings.kaggle_key)
    has_token_file = (Path.home() / ".kaggle" / "access_token").exists()
    has_legacy_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    ok = has_token or has_legacy or has_token_file or has_legacy_file
    detail = "found" if ok else "not found"
    fix = (
        "Set KAGGLE_API_TOKEN in .env, or save a token to ~/.kaggle/access_token "
        "(see README for setup)."
    )
    return CheckResult("Kaggle credentials", ok, detail, "" if ok else fix)


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


def check_environment(include_optional: bool = True) -> list[CheckResult]:
    """Run all environment checks and return their results, in report order."""
    results = [
        _check_python_version(),
        _check_lightgbm(),
        _check_kaggle_credentials(),
    ]
    if include_optional:
        results.extend([_check_image_deps(), _check_deep_deps()])
    return results


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
