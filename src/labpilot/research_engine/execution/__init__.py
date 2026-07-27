"""Execution Platform — Research Engineer + capability executors.

Owns plan-driven implementation: Engineer controller, capabilities, baseline
templates, codegen, training runner, and metrics used by generated pipelines.

Import hygiene: ``execution`` may import ``common``, ``accessor``, and planner
store/schema APIs. Prefer :class:`~labpilot.research_engine.intelligence.paths.ResearchPaths`
over deep intelligence imports.

Engineer symbols are lazy-imported so ``execution.baseline`` / stores can load
without pulling the full orchestrator (avoids cycles with ``experiments.graph``).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EngineerError",
    "ExecutionStore",
    "ResearchEngineer",
    "default_capability_registry",
    "default_stub_registry",
]


def __getattr__(name: str) -> Any:
    if name == "ExecutionStore":
        from labpilot.research_engine.execution.store import ExecutionStore

        return ExecutionStore
    if name in {
        "EngineerError",
        "ResearchEngineer",
        "default_capability_registry",
        "default_stub_registry",
    }:
        from labpilot.research_engine.execution import engineer as _engineer

        return getattr(_engineer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
