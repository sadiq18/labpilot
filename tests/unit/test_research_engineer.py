"""Unit tests for Research Engineer controller (Plan 2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.capabilities.stub import StubCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.engineer import (
    ResearchEngineer,
    default_stub_registry,
)
from labpilot.research_engine.execution.registry import CapabilityRegistry
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.models import (
    ResearchPlan,
    ResearchTask,
    RetryPolicy,
)
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore


def _seed_dag(knowledge: Path, competition: str = "demo") -> str:
    """A → B mini DAG."""
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="mini",
            status=PlanStatus.READY,
            tasks=[
                ResearchTask(
                    id="P-001-T01",
                    plan_id="P-001",
                    type=TaskType.PREPARE_WORKSPACE,
                    description="a",
                    order=0,
                ),
                ResearchTask(
                    id="P-001-T02",
                    plan_id="P-001",
                    type=TaskType.INSTALL_PACKAGE,
                    description="b",
                    dependencies=["P-001-T01"],
                    order=1,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        store.upsert_plan(plan)
        return plan.id
    finally:
        store.close()


def test_run_plan_stub_completes_dag(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_dag(knowledge)
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=default_stub_registry(),
    )
    try:
        execution = engineer.run_plan(plan_id)
        assert execution.status == "succeeded"
        assert execution.id == "E-001"
        plan = engineer._plan_store.get_plan(plan_id)
        assert plan is not None
        assert plan.status == PlanStatus.DONE
        assert all(t.status == TaskStatus.DONE for t in plan.tasks)
        # Evidence files written.
        ev = knowledge / "demo" / "research" / "executions" / "E-001" / "evidence"
        assert (ev / "P-001-T01.json").is_file()
        assert (ev / "P-001-T02.json").is_file()
    finally:
        engineer.close()


def test_resume_skips_done_tasks(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_dag(knowledge)
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=default_stub_registry(),
    )
    try:
        # Mark first task done, leave second pending; create partial execution.
        from labpilot.research_engine.execution.store import ExecutionStore

        store = PlanStore(knowledge, "demo")
        store.update_task_status("P-001-T01", TaskStatus.DONE)
        store.update_plan_status(plan_id, PlanStatus.IN_PROGRESS)
        store.close()

        exec_store = ExecutionStore(knowledge, "demo")
        execution = exec_store.create_execution(plan_id)
        exec_store.update_status(execution.id, "running")
        exec_store.close()

        result = engineer.resume(execution.id)
        assert result.status == "succeeded"
        plan = engineer._plan_store.get_plan(plan_id)
        assert plan is not None
        assert plan.tasks[0].status == TaskStatus.DONE
        assert plan.tasks[1].status == TaskStatus.DONE
    finally:
        engineer.close()


def test_failed_verify_fails_execution(tmp_path: Path) -> None:
    class FailCapability(BaseCapability):
        name = "fail"

        @property
        def supported_task_types(self):
            return frozenset(TaskType)

        def execute(self, context: TaskContext) -> TaskEvidence:
            return TaskEvidence(
                task_id=context.task.id,
                execution_id=context.execution.id,
                capability=self.name,
                passed=False,
                summary="boom",
                error="intentional fail",
            )

    knowledge = tmp_path / "knowledge"
    plan_id = _seed_dag(knowledge)
    # Give tasks zero retries so recovery fails immediately.
    store = PlanStore(knowledge, "demo")
    plan = store.get_plan(plan_id)
    assert plan is not None
    for task in plan.tasks:
        task.retry_policy = RetryPolicy(max_retries=0, abort_on_failure=True)
    store.upsert_plan(plan)
    store.close()

    registry = CapabilityRegistry()
    registry.register(FailCapability())
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=registry,
    )
    try:
        execution = engineer.run_plan(plan_id)
        assert execution.status == "failed"
        plan = engineer._plan_store.get_plan(plan_id)
        assert plan is not None
        assert plan.status == PlanStatus.ABANDONED
        assert plan.tasks[0].status == TaskStatus.FAILED
    finally:
        engineer.close()


def test_topo_dispatch_order(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_dag(knowledge)
    order: list[str] = []

    class OrderStub(StubCapability):
        def execute(self, context: TaskContext) -> TaskEvidence:
            order.append(context.task.id)
            return super().execute(context)

    registry = CapabilityRegistry()
    registry.register(OrderStub())
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=registry,
    )
    try:
        engineer.run_plan("P-001")
        assert order == ["P-001-T01", "P-001-T02"]
    finally:
        engineer.close()
