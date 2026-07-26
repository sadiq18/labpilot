"""Execution Platform — Research Engineer + capability executors.

Import hygiene: ``execution`` may import ``common``, ``accessor``, and planner
store/schema APIs. Prefer ``ResearchPaths`` over deep intelligence imports.
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
