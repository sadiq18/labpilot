"""Research Conductor — constrained LLM control plane over the tool catalog.

Import direction::

    CLI → conductor → {tools, artifacts, workspace_facade}
    engine packages must NOT import ``labpilot.research_engine.conductor``
"""

from __future__ import annotations

from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.models import (
    ApprovalResult,
    ConductSession,
    DecisionRecord,
    NextAction,
    Objective,
    OperatorFeedback,
    OsTask,
)
from labpilot.research_engine.conductor.store import ConductorStore

__all__ = [
    "ApprovalResult",
    "ConductSession",
    "ConductorStore",
    "DecisionRecord",
    "NextAction",
    "Objective",
    "OperatorFeedback",
    "OsTask",
    "run_until_stop",
]
