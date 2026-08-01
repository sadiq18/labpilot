"""Unit tests for ExecutionStore and evidence layout (Research Engineer Plan 1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from labpilot.accessor.sqlite import SCHEMA_VERSION, SqliteClient
from labpilot.research_engine.execution.evidence import read_evidence, write_evidence
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore


def _seed_plan(knowledge: Path, competition: str = "demo") -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="baseline",
            status=PlanStatus.READY,
            tasks=[
                ResearchTask(
                    id="P-001-T01",
                    plan_id="P-001",
                    type=TaskType.PREPARE_WORKSPACE,
                    description="prep",
                    status=TaskStatus.PENDING,
                )
            ],
            created_at=now,
            updated_at=now,
            metadata={"plan_kind": "baseline"},
        )
        store.upsert_plan(plan)
        return plan.id
    finally:
        store.close()


def test_migrate_adds_research_executions(tmp_path: Path) -> None:
    db = tmp_path / "knowledge.db"
    client = SqliteClient(db)
    try:
        assert client.schema_version() == SCHEMA_VERSION == "8"
        tables = {
            row["name"]
            for row in client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "research_executions" in tables
        assert "tasks" in tables  # Layer-3 untouched
    finally:
        client.close()


def test_execution_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_plan(knowledge)
    store = ExecutionStore(knowledge, "demo")
    try:
        execution = store.create_execution(plan_id)
        assert execution.id == "E-001"
        assert execution.plan_id == plan_id
        assert execution.status == "pending"
        workspace = Path(execution.workspace_path or "")
        assert workspace.is_dir()
        assert workspace.name == "demo"
        assert workspace.parent.name == "competitions"
        evidence_dir = (
            knowledge / "demo" / "research" / "executions" / "E-001" / "evidence"
        )
        assert evidence_dir.is_dir()

        store.update_status("E-001", "running")
        store.update_status("E-001", "succeeded")
        got = store.get_execution("E-001")
        assert got is not None
        assert got.status == "succeeded"
        assert got.started_at is not None
        assert got.completed_at is not None

        listed = store.list_executions(plan_id=plan_id)
        assert [e.id for e in listed] == ["E-001"]
    finally:
        store.close()


def test_create_execution_rejects_unknown_plan(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    # Ensure DB exists with schema but no plans.
    PlanStore(knowledge, "demo").close()
    store = ExecutionStore(knowledge, "demo")
    try:
        with pytest.raises(ValueError, match="unknown plan_id"):
            store.create_execution("P-999")
    finally:
        store.close()


def test_evidence_write_read(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    plan_id = _seed_plan(knowledge)
    store = ExecutionStore(knowledge, "demo")
    try:
        execution = store.create_execution(plan_id)
        evidence = TaskEvidence(
            task_id="P-001-T01",
            execution_id=execution.id,
            capability="workspace",
            passed=True,
            summary="dirs created",
            paths=["src", "configs"],
        )
        path = write_evidence(store.paths, evidence)
        assert path.is_file()
        loaded = read_evidence(store.paths, execution.id, "P-001-T01")
        assert loaded is not None
        assert loaded.summary == "dirs created"
        assert loaded.paths == ["src", "configs"]
    finally:
        store.close()


def test_plan_store_task_metadata_timing(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_plan(knowledge)
    store = PlanStore(knowledge, "demo")
    try:
        store.update_task_status("P-001-T01", TaskStatus.RUNNING)
        store.update_task_status(
            "P-001-T01", TaskStatus.DONE, metadata_patch={"capability": "stub"}
        )
        plan = store.get_plan("P-001")
        assert plan is not None
        task = plan.tasks[0]
        assert task.status == TaskStatus.DONE
        assert "started_at" in task.metadata
        assert "completed_at" in task.metadata
        assert task.metadata["capability"] == "stub"
    finally:
        store.close()
