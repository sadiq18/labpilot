"""Validate runtime credentials and configuration."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from labpilot.diagnostics import CheckResult, _check_kaggle_credentials
from labpilot.research_engine.execution.runtimes.models import (
    GoogleColabRuntime,
    KaggleKernelRuntime,
    LocalRuntime,
    OtherRuntime,
    RuntimeConfig,
)
from labpilot.research_engine.execution.runtimes.registry import load_runtimes


@dataclass
class RuntimeCheckResult:
    runtime_id: str
    provider: str
    ok: bool
    checks: list[CheckResult]


def _check_python_executable(path: str | None) -> CheckResult:
    executable = path or sys.executable
    exists = Path(executable).is_file() or shutil.which(executable) is not None
    return CheckResult(
        "Python executable",
        exists,
        executable,
        "" if exists else f"Python not found: {executable}",
    )


def _check_lightgbm_optional() -> CheckResult:
    try:
        import lightgbm  # noqa: F401
    except OSError as exc:
        return CheckResult("LightGBM import", False, str(exc), "Install libgomp / libomp runtime.")
    except ImportError as exc:
        return CheckResult("LightGBM import", False, str(exc), "Install lightgbm.")
    return CheckResult("LightGBM import", True, "ok", "")


def _check_env_var(name: str, label: str) -> CheckResult:
    value = os.environ.get(name, "").strip()
    ok = bool(value)
    return CheckResult(
        label,
        ok,
        "set" if ok else "missing",
        "" if ok else f"Set {name} in the environment.",
    )


def _check_kaggle_username(username: str | None) -> CheckResult:
    if username:
        return CheckResult("Kaggle username", True, username, "")
    return CheckResult(
        "Kaggle username",
        False,
        "not configured",
        "Set kaggle.username in runtime config or KAGGLE_USERNAME in .env.",
    )


def _check_other_adapter(runtime: OtherRuntime) -> CheckResult:
    module_path, _, attr = runtime.adapter.partition(":")
    if not module_path or not attr:
        return CheckResult(
            "Adapter module",
            False,
            runtime.adapter,
            "Adapter must be 'module.path:ClassName'.",
        )
    spec = importlib.util.find_spec(module_path)
    ok = spec is not None
    detail = "found" if ok else "not installed (execution deferred to P2)"
    return CheckResult("Adapter module", ok, detail, "" if ok else detail)


def check_runtime(
    runtime: RuntimeConfig,
    *,
    kaggle_username: str = "",
) -> RuntimeCheckResult:
    checks: list[CheckResult] = []

    if isinstance(runtime, LocalRuntime):
        checks.append(_check_python_executable(runtime.python))
        checks.append(_check_lightgbm_optional())
    elif isinstance(runtime, KaggleKernelRuntime):
        checks.append(_check_kaggle_credentials())
        checks.append(_check_kaggle_username(runtime.username or kaggle_username or None))
    elif isinstance(runtime, GoogleColabRuntime):
        checks.append(_check_env_var(runtime.auth.token_env, "Colab auth token"))
        if runtime.drive_sync is not None:
            checks.append(
                _check_env_var(
                    runtime.drive_sync.folder_id_env,
                    "Colab Drive folder",
                )
            )
    elif isinstance(runtime, OtherRuntime):
        checks.append(_check_other_adapter(runtime))
        if runtime.key_env:
            checks.append(_check_env_var(runtime.key_env, "SSH key path"))

    ok = all(check.ok for check in checks) if checks else True
    return RuntimeCheckResult(runtime.id, runtime.provider, ok, checks)


def check_all_runtimes(
    *directories: Path,
    runtime_id: str | None = None,
    kaggle_username: str = "",
) -> list[RuntimeCheckResult]:
    runtimes = load_runtimes(*directories)
    if runtime_id is not None:
        runtime = runtimes.get(runtime_id)
        if runtime is None:
            raise ValueError(f"Runtime not found: {runtime_id}")
        return [check_runtime(runtime, kaggle_username=kaggle_username)]

    return [
        check_runtime(runtime, kaggle_username=kaggle_username)
        for runtime in sorted(runtimes.values(), key=lambda item: item.id)
    ]
