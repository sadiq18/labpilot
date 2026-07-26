"""Runtime registry and validation (config-only in v1.0)."""

from labpilot.research_engine.execution.runtimes.doctor import RuntimeCheckResult, check_all_runtimes, check_runtime
from labpilot.research_engine.execution.runtimes.models import RuntimeConfig, RuntimeRecord
from labpilot.research_engine.execution.runtimes.registry import get_runtime, list_runtimes, load_runtimes
from labpilot.research_engine.execution.runtimes.templates import runtime_to_yaml_dict, scaffold_runtime

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
