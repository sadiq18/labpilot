"""Research Planner — the planning *compiler*.

Turns a Hypothesis into a validated, database-backed executable DAG
(``ResearchPlan`` of ``ResearchTask`` nodes). Deterministic Python owns the
pipeline (retrieval, context, templates, validation, scheduling, persistence);
the LLM is one optional Planning Engine stage. The planner never writes code or
runs training — it only emits plan nodes.
"""

from labpilot.research_engine.planner.planner import (
    BaselinePlanError,
    blueprint_to_draft,
    compile_baseline_plan,
    compile_research_plan,
    lower_draft,
)
from labpilot.research_engine.planner.schemas import (
    DraftTask,
    PlanStatus,
    ResearchPlan,
    ResearchPlanDraft,
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
    "BaselinePlanError",
    "blueprint_to_draft",
    "compile_baseline_plan",
    "compile_research_plan",
    "lower_draft",
    "DraftTask",
    "ResearchPlanDraft",
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
