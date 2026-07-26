"""Execution Platform — Research Engineer + capability executors.

Owns plan-driven implementation: Engineer controller, capabilities, baseline
templates, codegen, training runner, and metrics used by generated pipelines.

Import hygiene: ``execution`` may import ``common``, ``accessor``, and planner
store/schema APIs. Prefer :class:`~labpilot.research_engine.intelligence.paths.ResearchPaths`
over deep intelligence imports.
"""

from labpilot.research_engine.execution.engineer import (
    EngineerError,
    ResearchEngineer,
    default_capability_registry,
    default_stub_registry,
)
from labpilot.research_engine.execution.store import ExecutionStore

__all__ = [
    "EngineerError",
    "ExecutionStore",
    "ResearchEngineer",
    "default_capability_registry",
    "default_stub_registry",
]
