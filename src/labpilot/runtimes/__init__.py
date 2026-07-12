"""Runtime registry and validation (config-only in v1.0)."""

from labpilot.runtimes.doctor import RuntimeCheckResult, check_all_runtimes, check_runtime
from labpilot.runtimes.models import RuntimeConfig, RuntimeRecord
from labpilot.runtimes.registry import get_runtime, list_runtimes, load_runtimes
from labpilot.runtimes.templates import runtime_to_yaml_dict, scaffold_runtime

__all__ = [
    "RuntimeCheckResult",
    "RuntimeConfig",
    "RuntimeRecord",
    "check_all_runtimes",
    "check_runtime",
    "get_runtime",
    "list_runtimes",
    "load_runtimes",
    "runtime_to_yaml_dict",
    "scaffold_runtime",
]
