"""The planner instruction set and lifecycle enums.

The Planning Engine chooses among these ~15 typed instructions; a future
executor dispatches on them. The planner itself **never** performs the side
effect — a task is a node describing intent, not an action.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Instruction set for research task DAG nodes."""

    PREPARE_WORKSPACE = "prepare_workspace"
    READ_CODE = "read_code"
    WRITE_CODE = "write_code"
    MODIFY_CONFIG = "modify_config"
    RESEARCH_REVIEW = "research_review"
    INSTALL_PACKAGE = "install_package"
    RUN_UNIT_TEST = "run_unit_test"
    RUN_SMOKE_TEST = "run_smoke_test"
    SELECT_RUNTIME = "select_runtime"
    RUN_TRAINING = "run_training"
    RUN_INFERENCE = "run_inference"
    BUILD_SUBMISSION = "build_submission"
    EVALUATE = "evaluate"
    COMPARE = "compare"
    GENERATE_REPORT = "generate_report"
    UPDATE_BELIEF = "update_belief"
    CREATE_HYPOTHESIS = "create_hypothesis"
    REFLECT = "reflect"


class PlanStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ABANDONED = "abandoned"


#: Statuses the Engineer will actually accept as a run target.
#:
#: `Engineer._load_plan` raises "need ready or in_progress" for anything else,
#: and reopens ABANDONED to READY before that check — so an abandoned plan is
#: runnable and a done one is not. This constant exists because three call sites
#: had each written their own version of this set and none of them matched:
#: `_latest_plan_id` used {ready, in_progress, draft}, `has_unrun_plan` used
#: {ready, draft}, and the Engineer used {ready, in_progress}. The Conductor
#: therefore offered `run_plan` for finished plans and lost a step each time.
RUNNABLE_PLAN_STATUSES: frozenset[PlanStatus] = frozenset(
    {PlanStatus.READY, PlanStatus.IN_PROGRESS, PlanStatus.ABANDONED}
)


#: Statuses meaning "this plan has not produced a result yet".
#:
#: Deliberately *wider* than RUNNABLE: a DRAFT cannot be dispatched but is still
#: outstanding work, so queuing another plan on top of it starves the one that
#: exists. The two sets answer different questions and collapsing them is what
#: produced the bug — "can I run this?" is not "is there work pending?".
UNRUN_PLAN_STATUSES: frozenset[PlanStatus] = frozenset(
    {PlanStatus.DRAFT, PlanStatus.READY, PlanStatus.IN_PROGRESS, PlanStatus.ABANDONED}
)


def is_runnable_plan_status(status: object) -> bool:
    """Whether a plan in this status can still be dispatched to the Engineer."""
    try:
        return PlanStatus(str(status)) in RUNNABLE_PLAN_STATUSES
    except ValueError:
        return False


def is_unrun_plan_status(status: object) -> bool:
    """Whether a plan in this status still represents outstanding work."""
    try:
        return PlanStatus(str(status)) in UNRUN_PLAN_STATUSES
    except ValueError:
        return False


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class RuntimeTarget(StrEnum):
    """Where a task/plan is expected to run — a hook for a future planner."""

    LOCAL = "local"
    DOCKER = "docker"
    KAGGLE = "kaggle"
    CPU = "cpu"
    P100 = "p100"
    A100 = "a100"
