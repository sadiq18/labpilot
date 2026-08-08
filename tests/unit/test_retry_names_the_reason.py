"""A rebuild must tell codegen why the last file failed.

Re-queuing `write_code` without the reason asks the model to try again from the
inputs that produced the broken file — it makes the same mistake and costs a
step doing it. `AGENTS.md` records the opposite working: naming the failure took
prose-reply failures from three-in-eight to 30 of 30.

Measured on rogii 2026-08-08. Infrastructure was finally sound — dependencies
resolved, uv built an environment, LightGBM ran — and the script died on
``ValueError: pandas dtypes must be int, float or bool. Fields with bad pandas
dtypes: Geology: object``. A blind rebuild would hand LightGBM the same string
column.
"""

from __future__ import annotations

from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)

_NOW = "2026-08-08T00:00:00+00:00"
_GEOLOGY = (
    "ValueError: pandas dtypes must be int, float or bool.\n"
    "Fields with bad pandas dtypes: Geology: object"
)


class _FakeStore:
    def __init__(self) -> None:
        self.patches: dict[str, dict] = {}

    def update_task_status(self, task_id, status, *, metadata_patch=None, error=None):
        self.patches[task_id] = dict(metadata_patch or {})


def _task(tid, ttype, status, deps, *, error="", order=0) -> ResearchTask:
    return ResearchTask(
        id=tid,
        plan_id="P-021",
        type=ttype,
        description=tid,
        status=status,
        dependencies=deps,
        metadata={"error": error} if error else {},
        order=order,
    )


def _plan(*tasks) -> ResearchPlan:
    return ResearchPlan(
        id="P-021",
        competition="rogii-wellbore-geology-prediction",
        hypothesis_id="H-051",
        goal="ensemble",
        status=PlanStatus.ABANDONED,
        created_at=_NOW,
        updated_at=_NOW,
        tasks=list(tasks),
    )


def _reset(plan, *, unrunnable=False) -> dict[str, dict]:
    from labpilot.research_engine.execution.engineer import ResearchEngineer

    store = _FakeStore()
    engineer = ResearchEngineer.__new__(ResearchEngineer)
    engineer._plan_store = store  # type: ignore[attr-defined]
    engineer._train_script_is_unrunnable = lambda: unrunnable  # type: ignore[method-assign]
    engineer._reset_tasks_for_retry(plan)
    return store.patches


def _standard_plan(smoke_error: str = _GEOLOGY) -> ResearchPlan:
    return _plan(
        _task("T02", TaskType.WRITE_CODE, TaskStatus.DONE, [], order=1),
        _task(
            "T04", TaskType.RUN_SMOKE_TEST, TaskStatus.FAILED, ["T02"], error=smoke_error, order=2
        ),
    )


def test_the_rebuild_carries_the_failure_that_caused_it():
    patches = _reset(_standard_plan())

    assert "Geology: object" in patches["T02"]["retry_reason"]


def test_a_code_failure_is_preferred_over_a_downstream_one():
    """A smoke test names what would not run; `evaluate` failing later
    describes a consequence."""
    plan = _plan(
        _task("T02", TaskType.WRITE_CODE, TaskStatus.DONE, [], order=1),
        _task("T04", TaskType.RUN_SMOKE_TEST, TaskStatus.FAILED, ["T02"], error=_GEOLOGY, order=2),
        _task("T06", TaskType.EVALUATE, TaskStatus.FAILED, ["T04"], error="no metrics", order=3),
    )

    assert "Geology" in _reset(plan)["T02"]["retry_reason"]


def test_only_the_code_task_is_told_why():
    """`run_smoke_test` does not need the reason — it is not what gets rewritten."""
    patches = _reset(_standard_plan())

    assert "retry_reason" not in patches.get("T04", {})


def test_no_reason_is_invented_when_the_failure_recorded_none():
    patches = _reset(_standard_plan(smoke_error=""))

    assert "retry_reason" not in patches["T02"]


def test_the_retry_marker_is_still_set():
    """The carve-out must not cost the behaviour it guards."""
    patches = _reset(_standard_plan())

    assert patches["T02"]["retried_after_abandon"] is True
