"""Load and merge runtime YAML definitions."""

from __future__ import annotations

from pathlib import Path

import yaml

from labpilot.research_engine.execution.runtimes.models import (
    GoogleColabRuntime,
    KaggleKernelRuntime,
    LocalRuntime,
    OtherRuntime,
    RuntimeConfig,
)

BUILTIN_LOCAL_DEFAULT = LocalRuntime(id="local-default")


def _iter_runtime_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name.startswith("."):
            continue
        files.append(path)
    return files


def _parse_runtime(raw: dict) -> RuntimeConfig:
    provider = raw.get("provider", "local")
    if provider == "local":
        return LocalRuntime.model_validate(raw)
    if provider == "kaggle_kernel":
        return KaggleKernelRuntime.model_validate(raw)
    if provider == "google_colab":
        return GoogleColabRuntime.model_validate(raw)
    if provider == "other":
        return OtherRuntime.model_validate(raw)
    raise ValueError(f"Unknown runtime provider: {provider!r}")


def load_runtimes(
    *directories: Path,
    include_builtin: bool = True,
) -> dict[str, RuntimeConfig]:
    """Load runtime configs from one or more directories; later dirs override by id."""
    runtimes: dict[str, RuntimeConfig] = {}
    if include_builtin:
        runtimes[BUILTIN_LOCAL_DEFAULT.id] = BUILTIN_LOCAL_DEFAULT

    for directory in directories:
        for path in _iter_runtime_files(directory):
            raw = yaml.safe_load(path.read_text()) or {}
            if "id" not in raw:
                raw["id"] = path.stem
            runtime = _parse_runtime(raw)
            runtimes[runtime.id] = runtime

    return runtimes


def get_runtime(
    runtime_id: str,
    *directories: Path,
) -> RuntimeConfig | None:
    return load_runtimes(*directories).get(runtime_id)


def list_runtimes(
    *directories: Path,
    enabled_only: bool = False,
) -> list[RuntimeConfig]:
    runtimes = load_runtimes(*directories)
    items = sorted(runtimes.values(), key=lambda item: (-item.priority, item.id))
    if enabled_only:
        items = [item for item in items if item.enabled]
    return items
