"""Research Conductor — constrained LLM control plane over the tool catalog.

Import direction::

    CLI → conductor → {tools, artifacts, workspace_facade}
    engine packages must NOT import ``labpilot.research_engine.conductor``
"""

from __future__ import annotations

from labpilot.research_engine.conductor.actions import (
    ActionPlan,
    ResearchAction,
    map_research_action,
)
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    evaluate_stops,
)
from labpilot.research_engine.conductor.checkpoint import latest_active_session, save_checkpoint
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.metrics import CampaignMetrics, Suggestion
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
    "ActionPlan",
    "ApprovalResult",
    "BudgetConfig",
    "BudgetState",
    "CampaignMetrics",
    "ConductSession",
    "ConductorStore",
    "DecisionRecord",
    "NextAction",
    "Objective",
    "OperatorFeedback",
    "OsTask",
    "ResearchAction",
    "Suggestion",
    "evaluate_stops",
    "latest_active_session",
    "map_research_action",
    "run_until_stop",
    "save_checkpoint",
]
