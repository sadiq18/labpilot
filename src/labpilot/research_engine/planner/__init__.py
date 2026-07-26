"""Research Planner — the planning *compiler*.

Turns a Hypothesis into a validated, database-backed executable DAG
(``ResearchPlan`` of ``ResearchTask`` nodes). Deterministic Python owns the
pipeline (retrieval, context, templates, validation, scheduling, persistence);
the LLM is one optional stage (Plan 4). The planner never writes code or runs
training — it only emits plan nodes.
"""

from labpilot.research_engine.planner.planner import compile_research_plan
from labpilot.research_engine.planner.schemas import (
    PlanStatus,
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
    RuntimeTarget,
    TaskStatus,
    TaskType,
    TaskVerification,
)
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import (
    PlanValidationError,
    topological_levels,
    validate_plan,
)

__all__ = [
    "compile_research_plan",
    "PlanStore",
    "PlanValidationError",
    "PlanStatus",
    "ResearchPlan",
    "ResearchTask",
    "RetryPolicy",
    "RuntimeTarget",
    "TaskStatus",
    "TaskType",
    "TaskVerification",
    "topological_levels",
    "validate_plan",
]
