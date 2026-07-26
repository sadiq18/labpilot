"""Tests for Workspace + Dependency capabilities (Plan 4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.capabilities.dependency import (
    DependencyCapability,
)
from labpilot.research_engine.execution.capabilities.workspace import (
    WorkspaceCapability,
    default_workspace_dirs,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.engineer import (
    ResearchEngineer,
    default_capability_registry,
)
from labpilot.research_engine.execution.evidence import read_evidence
from labpilot.research_engine.execution.registry import CapabilityRegistry
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore


def _seed_workspace_deps_plan(knowledge: Path, competition: str = "demo") -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="workspace+deps",
            status=PlanStatus.READY,
            tasks=[
                ResearchTask(
                    id="P-001-T01",
                    plan_id="P-001",
                    type=TaskType.PREPARE_WORKSPACE,
                    description="prep workspace",
                    order=0,
                ),
                ResearchTask(
                    id="P-001-T02",
                    plan_id="P-001",
                    type=TaskType.INSTALL_PACKAGE,
                    description="install deps",
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


def test_workspace_creates_expected_tree(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_workspace_deps_plan(knowledge)
    paths = ResearchPaths(knowledge, "demo").ensure()
    root = tmp_path / "competitions" / "demo"
    store = PlanStore(knowledge, "demo")
    plan = store.get_plan(plan_id)
    assert plan is not None
    task = plan.tasks[0]
    store.close()

    execution = ResearchExecution(id="E-test", plan_id=plan_id, competition="demo")
    context = TaskContext(
        plan=plan,
        task=task,
        execution=execution,
        paths=paths,
        workspace_root=root,
        competition="demo",
    )
    cap = WorkspaceCapability()
    first = cap.execute(context)
    assert first.passed
    assert first.metadata.get("idempotent") is False
    assert Path(first.metadata["workspace"]).name == "demo"
    for path in default_workspace_dirs(root):
        assert path.is_dir()

    second = cap.execute(context)
    assert second.passed
    assert second.metadata.get("idempotent") is True


def test_dependency_noop_when_satisfied(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_workspace_deps_plan(knowledge)
    paths = ResearchPaths(knowledge, "demo").ensure()
    (paths.root / "requirements.txt").write_text("# none\n", encoding="utf-8")

    registry = default_capability_registry(install_packages=False)
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=registry,
    )
    try:
        execution = engineer.run_plan(plan_id)
        assert execution.status == "succeeded"
        ev = read_evidence(engineer.paths, execution.id, "P-001-T02")
        assert ev is not None
        assert ev.passed
        assert (
            ev.metadata.get("skipped")
            or ev.metadata.get("idempotent")
            or "satisfied" in ev.summary
            or "skipped" in ev.summary
            or "no requirements" in ev.summary
        )
    finally:
        engineer.close()


def test_engineer_workspace_deps_integration(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_workspace_deps_plan(knowledge)
    engineer = ResearchEngineer(
        knowledge_dir=knowledge,
        competition="demo",
        registry=default_capability_registry(install_packages=False),
    )
    try:
        execution = engineer.run_plan(plan_id)
        assert execution.status == "succeeded"
        workspace = Path(execution.workspace_path or "")
        assert workspace.name == "demo"
        assert workspace.parent.name == "competitions"
        assert (workspace / "src").is_dir()
        plan = engineer._plan_store.get_plan(plan_id)
        assert plan is not None
        assert [t.status for t in plan.tasks] == [TaskStatus.DONE, TaskStatus.DONE]
    finally:
        engineer.close()
