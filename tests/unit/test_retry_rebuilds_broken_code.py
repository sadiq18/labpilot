"""A retry must not re-run the file that failed the gate.

Measured on rogii 2026-08-08. `pipeline/train.py` imported `catboost` with no
PEP 723 declaration, so the smoke test failed and the plan was abandoned. Every
retry reopened the plan and resumed *after* `write_code`, which was `done` —
re-executing the identical file. Sixteen `run_experiment` dispatches, zero
`write_code`.

The fix for the underlying defect (generated code declaring its dependencies)
had shipped eight days earlier. It could never be reached, because the only step
that would have applied it was already marked complete.
"""

from __future__ import annotations

from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
)
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)

_NOW = "2026-08-08T00:00:00+00:00"


class _FakeStore:
    """Records status transitions instead of touching SQLite."""

    def __init__(self) -> None:
        self.updates: dict[str, TaskStatus] = {}

    def update_task_status(self, task_id, status, *, metadata_patch=None, error=None):
        self.updates[task_id] = status


def _task(tid: str, ttype: TaskType, status: TaskStatus, deps: list[str]) -> ResearchTask:
    return ResearchTask(
        id=tid,
        plan_id="P-019",
        type=ttype,
        description=tid,
        status=status,
        dependencies=deps,
    )


def _plan(smoke_status: TaskStatus, *, training_status=TaskStatus.PENDING) -> ResearchPlan:
    return ResearchPlan(
        id="P-019",
        competition="rogii-wellbore-geology-prediction",
        hypothesis_id="H-019",
        goal="ensemble",
        status=PlanStatus.ABANDONED,
        created_at=_NOW,
        updated_at=_NOW,
        tasks=[
            _task("T01", TaskType.READ_CODE, TaskStatus.DONE, []),
            _task("T02", TaskType.WRITE_CODE, TaskStatus.DONE, ["T01"]),
            _task("T04", TaskType.RUN_SMOKE_TEST, smoke_status, ["T02"]),
            _task("T05", TaskType.RUN_TRAINING, training_status, ["T04"]),
        ],
    )


def _reset(plan: ResearchPlan) -> dict[str, TaskStatus]:
    from labpilot.research_engine.execution.engineer import ResearchEngineer

    store = _FakeStore()
    engineer = ResearchEngineer.__new__(ResearchEngineer)
    engineer._plan_store = store  # type: ignore[attr-defined]
    engineer._reset_tasks_for_retry(plan)
    return store.updates


def test_a_failed_smoke_test_rebuilds_the_code():
    """The exact rogii loop: without this, retry re-runs the same broken file."""
    updates = _reset(_plan(TaskStatus.FAILED))

    assert updates.get("T02") == TaskStatus.PENDING, "write_code must be re-queued"
    assert updates.get("T04") == TaskStatus.PENDING


def test_a_failed_unit_test_also_rebuilds_the_code():
    plan = _plan(TaskStatus.DONE)
    plan.tasks[2] = _task("T04", TaskType.RUN_UNIT_TEST, TaskStatus.FAILED, ["T02"])

    assert _reset(plan).get("T02") == TaskStatus.PENDING


def test_a_failed_training_run_does_not_discard_working_code():
    """`run_training` fails for reasons code cannot fix — a missing dataset, an
    OOM. Regenerating then would throw away a file that passed its gates and
    spend a codegen call to do it."""
    plan = _plan(TaskStatus.DONE, training_status=TaskStatus.FAILED)

    updates = _reset(plan)
    assert updates.get("T05") == TaskStatus.PENDING
    assert "T02" not in updates, "write_code must survive a non-code failure"


def test_nothing_is_reset_when_nothing_failed():
    assert _reset(_plan(TaskStatus.DONE)) == {}
