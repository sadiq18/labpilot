"""Unit tests for EvidenceExtractor (Milestone 6 Plan 2)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.experiments.models import ExperimentComparison, Verdict
from labpilot.research_engine.execution.evidence import ensure_execution_layout, write_evidence
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    TaskStatus,
    TaskType,
)
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.reflection.evidence import EvidenceExtractor, assess_strength


def test_assess_strength_rules() -> None:
    assert assess_strength(execution_failed=True) == "rejected"
    assert (
        assess_strength(comparison={"verdict": Verdict.REGRESSION.value}) == "rejected"
    )
    assert (
        assess_strength(
            comparison={"delta": 0.02, "verdict": Verdict.WORTH_KEEPING.value}
        )
        == "strong"
    )
    assert assess_strength(comparison={"delta": 0.0005}) == "weak"
    assert assess_strength(comparison={"outcome": "baseline"}, metrics={"cv": 0.5}) == (
        "moderate"
    )
    assert assess_strength(metrics={}) == "weak"


def test_extract_from_workspace_fixture_persists(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    workspace = tmp_path / "competitions" / "demo"
    workspace.mkdir(parents=True)
    (workspace / "metrics.json").write_text(
        json.dumps({"cv_accuracy": 0.82, "runtime_seconds": 12.5}),
        encoding="utf-8",
    )
    (workspace / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_classification",
                "template_name": "tabular_classification",
                "metric_name": "accuracy",
            }
        ),
        encoding="utf-8",
    )
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "comparison.json").write_text(
        json.dumps(
            {
                "compare_to": "P-001",
                "delta": 0.015,
                "verdict": "worth_keeping",
                "maximize": True,
            }
        ),
        encoding="utf-8",
    )

    extractor = EvidenceExtractor(knowledge, "demo")
    try:
        evidence = extractor.extract(
            execution_id="E-001",
            workspace_path=workspace,
            plan_id="P-002",
            experiment_id="exp_demo_E-001",
        )
        assert evidence["id"] == "EE-001"
        assert evidence["strength"] == "strong"
        assert evidence["metrics"]["cv_accuracy"] == 0.82
        assert evidence["config_summary"]["baseline_choice"]["template_name"] == (
            "tabular_classification"
        )
        assert evidence["runtime_summary"]["runtime_seconds"] == 12.5
        assert evidence["comparison"]["delta"] == 0.015
        assert evidence["plan_id"] == "P-002"

        again = extractor.extract(
            workspace_path=workspace,
            comparison={"delta": 0.015},
            persist=False,
        )
        assert "id" not in again
        assert again["strength"] == "strong"
    finally:
        extractor.close()


def test_extract_failed_execution_is_rejected(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metrics.json").write_text("{}", encoding="utf-8")

    extractor = EvidenceExtractor(knowledge, "demo")
    try:
        evidence = extractor.extract(
            workspace_path=workspace,
            execution_status="failed",
            execution_error="train OOM",
        )
        assert evidence["strength"] == "rejected"
        assert evidence["metadata"]["execution_error"] == "train OOM"
    finally:
        extractor.close()


def test_extract_uses_experiment_comparison_model(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "metrics.json").write_text(
        json.dumps({"cv_accuracy": 0.7}), encoding="utf-8"
    )
    comparison = ExperimentComparison(
        base_id="base",
        compare_id="child",
        primary_metric_key="cv_accuracy",
        metric_deltas={"cv_accuracy": 0.0002},
        changes=[],
        runtime_delta_seconds=1.0,
        runtime_delta_pct=5.0,
        verdict=Verdict.INCONCLUSIVE,
        verdict_reason="noise",
    )
    extractor = EvidenceExtractor(knowledge, "demo")
    try:
        evidence = extractor.extract(
            workspace_path=workspace,
            comparison=comparison,
        )
        assert evidence["strength"] == "weak"
        assert evidence["comparison"]["verdict"] == "inconclusive"
    finally:
        extractor.close()


def test_extract_resolves_execution_and_task_failures(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    competition = "demo"
    _seed_plan(knowledge, competition)
    store = ExecutionStore(knowledge, competition)
    try:
        execution = store.create_execution("P-001")
        store.update_status(execution.id, "failed", error="smoke failed")
        workspace = Path(execution.workspace_path or "")
        (workspace / "metrics.json").write_text(
            json.dumps({"cv_accuracy": 0.4}), encoding="utf-8"
        )
        paths = ResearchPaths(knowledge, competition).ensure()
        ensure_execution_layout(paths, execution.id)
        write_evidence(
            paths,
            TaskEvidence(
                execution_id=execution.id,
                task_id="P-001-T01",
                capability="verification",
                passed=False,
                summary="smoke failed",
                created_at=datetime.now(UTC),
            ),
        )
    finally:
        store.close()

    extractor = EvidenceExtractor(knowledge, competition)
    try:
        evidence = extractor.extract(execution_id=execution.id)
        assert evidence["execution_id"] == execution.id
        assert evidence["plan_id"] == "P-001"
        assert evidence["strength"] == "rejected"
        assert evidence["metadata"]["task_evidence"]["has_failure"] is True
    finally:
        extractor.close()


def _seed_plan(knowledge: Path, competition: str) -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="H-001",
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
