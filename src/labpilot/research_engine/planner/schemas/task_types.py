"""The planner instruction set and lifecycle enums.

The Planning Engine chooses among these ~15 typed instructions; a future
executor dispatches on them. The planner itself **never** performs the side
effect — a task is a node describing intent, not an action.
"""

from __future__ import annotations

from enum import StrEnum


class TaskType(StrEnum):
    """Instruction set for research task DAG nodes."""

    READ_CODE = "read_code"
    WRITE_CODE = "write_code"
    MODIFY_CONFIG = "modify_config"
    INSTALL_PACKAGE = "install_package"
    RUN_UNIT_TEST = "run_unit_test"
    RUN_SMOKE_TEST = "run_smoke_test"
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
