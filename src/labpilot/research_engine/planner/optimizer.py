"""Deterministic post-processing: fill per-type verification/retry defaults.

Runs after lowering the template (and, later, after the Planning Engine draft):
any task left without a verification ``check`` gets a sensible type default, and
training-like tasks get a default retry budget. This is the "deterministic merge"
step — no LLM, no side effects.
"""

from __future__ import annotations

from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    RetryPolicy,
    TaskVerification,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

_DEFAULT_VERIFICATION: dict[TaskType, TaskVerification] = {
    TaskType.PREPARE_WORKSPACE: TaskVerification(
        check="Workspace directories exist and are writable.",
        failure_recovery="Recreate missing directories.",
    ),
    TaskType.READ_CODE: TaskVerification(check="Relevant code located."),
    TaskType.WRITE_CODE: TaskVerification(
        check="Change implemented.",
        failure_recovery="Revert changes via git.",
    ),
    TaskType.MODIFY_CONFIG: TaskVerification(
        check="Config loads successfully.",
        failure_recovery="Restore previous config version.",
    ),
    TaskType.RESEARCH_REVIEW: TaskVerification(
        check="No critical research-correctness findings.",
        failure_recovery="Revise code via WRITE_CODE, or abort.",
    ),
    TaskType.INSTALL_PACKAGE: TaskVerification(
        check="Package installs and imports.",
        failure_recovery="Pin/roll back the dependency version.",
    ),
    TaskType.RUN_UNIT_TEST: TaskVerification(
        check="Exit 0; required tests pass.",
        failure_recovery="Fix via a WRITE_CODE task, or abort.",
    ),
    TaskType.RUN_SMOKE_TEST: TaskVerification(
        check="Runs end-to-end on a tiny sample without error.",
        failure_recovery="Fix via WRITE_CODE/MODIFY_CONFIG, or abort.",
    ),
    TaskType.SELECT_RUNTIME: TaskVerification(
        check="Runtime target selected and recorded.",
        failure_recovery="Fall back to local runtime, or abort.",
    ),
    TaskType.RUN_TRAINING: TaskVerification(
        check="Loss decreases (or metrics are finite).",
        failure_recovery="Abort after N failures.",
    ),
    TaskType.RUN_INFERENCE: TaskVerification(
        check="Predictions produced for the expected inputs.",
        failure_recovery="Retry; abort after N failures.",
    ),
    TaskType.BUILD_SUBMISSION: TaskVerification(
        check="Submission artifact matches the expected format.",
        failure_recovery="Regenerate from the latest run.",
    ),
    TaskType.EVALUATE: TaskVerification(check="Metrics recorded."),
    TaskType.COMPARE: TaskVerification(
        check="Metric delta recorded vs baseline.",
        failure_recovery="Mark inconclusive.",
    ),
    TaskType.GENERATE_REPORT: TaskVerification(check="Report written."),
    TaskType.UPDATE_BELIEF: TaskVerification(check="Belief store updated."),
    TaskType.CREATE_HYPOTHESIS: TaskVerification(check="New hypothesis recorded."),
    TaskType.REFLECT: TaskVerification(check="Reflection captured."),
}

#: Task types that get a retry budget by default (transient/flaky work).
_DEFAULT_RETRIES: dict[TaskType, int] = {
    TaskType.RUN_TRAINING: 2,
    TaskType.RUN_INFERENCE: 1,
    TaskType.INSTALL_PACKAGE: 1,
}


def _is_empty(verification: TaskVerification) -> bool:
    return not (
        verification.check
        or verification.expected_output
        or verification.failure_recovery
    )


def apply_defaults(plan: ResearchPlan) -> ResearchPlan:
    """Fill missing verification/retry defaults in place; returns ``plan``."""
    for task in plan.tasks:
        if _is_empty(task.verification):
            default = _DEFAULT_VERIFICATION.get(task.type)
            if default is not None:
                task.verification = default.model_copy()
        if task.retry_policy == RetryPolicy() and task.type in _DEFAULT_RETRIES:
            task.retry_policy = RetryPolicy(
                max_retries=_DEFAULT_RETRIES[task.type],
                abort_on_failure=True,
            )
    return plan
