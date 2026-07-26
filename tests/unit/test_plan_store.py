from datetime import UTC, datetime
from pathlib import Path

import pytest

from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
)
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import (
    PlanValidationError,
    topological_levels,
    validate_plan,
)


def _plan(plan_id: str = "P-001") -> ResearchPlan:
    now = datetime.now(UTC)
    tasks = [
        ResearchTask(id=f"{plan_id}-T01", plan_id=plan_id, type=TaskType.READ_CODE),
        ResearchTask(
            id=f"{plan_id}-T02",
            plan_id=plan_id,
            type=TaskType.WRITE_CODE,
            dependencies=[f"{plan_id}-T01"],
        ),
        ResearchTask(
            id=f"{plan_id}-T03",
            plan_id=plan_id,
            type=TaskType.RUN_UNIT_TEST,
            dependencies=[f"{plan_id}-T02"],
        ),
    ]
    return ResearchPlan(
        id=plan_id,
        competition="demo",
        hypothesis_id="H-001",
        goal="test goal",
        status=PlanStatus.READY,
        tasks=tasks,
        created_at=now,
        updated_at=now,
    )


def test_upsert_then_get_round_trips_tasks_and_edges(tmp_path: Path):
    store = PlanStore(tmp_path / "knowledge", "demo")
    try:
        store.upsert_plan(_plan())
        got = store.get_plan("P-001")
        assert got is not None
        assert [t.id for t in got.tasks] == ["P-001-T01", "P-001-T02", "P-001-T03"]
        assert got.tasks[1].dependencies == ["P-001-T01"]
        assert got.tasks[2].dependencies == ["P-001-T02"]
        assert got.status == PlanStatus.READY
    finally:
        store.close()


def test_new_plan_id_increments(tmp_path: Path):
    store = PlanStore(tmp_path / "knowledge", "demo")
    try:
        assert store.new_plan_id() == "P-001"
        store.upsert_plan(_plan("P-001"))
        assert store.new_plan_id() == "P-002"
    finally:
        store.close()


def test_upsert_replaces_tasks(tmp_path: Path):
    store = PlanStore(tmp_path / "knowledge", "demo")
    try:
        store.upsert_plan(_plan())
        smaller = _plan()
        smaller.tasks = smaller.tasks[:1]
        store.upsert_plan(smaller)
        got = store.get_plan("P-001")
        assert got is not None
        assert len(got.tasks) == 1
        # Cascade removed dependency edges for the dropped tasks.
        remaining = store._conn.execute(
            "SELECT COUNT(*) AS n FROM research_task_deps"
        ).fetchone()["n"]
        assert remaining == 0
    finally:
        store.close()


def test_list_and_status_updates(tmp_path: Path):
    store = PlanStore(tmp_path / "knowledge", "demo")
    try:
        store.upsert_plan(_plan())
        assert [p.id for p in store.list_plans(hypothesis_id="H-001")] == ["P-001"]
        store.update_plan_status("P-001", PlanStatus.DONE)
        store.update_task_status("P-001-T01", TaskStatus.DONE)
        got = store.get_plan("P-001")
        assert got.status == PlanStatus.DONE
        assert got.tasks[0].status == TaskStatus.DONE
    finally:
        store.close()


def test_validator_accepts_valid_dag():
    plan = _plan()
    validate_plan(plan)
    assert topological_levels(plan) == [["P-001-T01"], ["P-001-T02"], ["P-001-T03"]]


def test_validator_rejects_missing_dependency():
    plan = _plan()
    plan.tasks[0].dependencies = ["P-001-T99"]
    with pytest.raises(PlanValidationError):
        validate_plan(plan)


def test_validator_rejects_cycle():
    plan = _plan()
    plan.tasks[0].dependencies = ["P-001-T03"]  # T01 <- T03, T03 <- T02 <- T01
    with pytest.raises(PlanValidationError):
        validate_plan(plan)
