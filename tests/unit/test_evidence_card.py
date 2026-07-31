"""Evidence Card, COMPARE wiring, attribution, and Research Graph."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.evidence.attribution import attribute_techniques
from labpilot.research_engine.evidence.builder import (
    build_evidence_card,
    write_comparison_files,
)
from labpilot.research_engine.evidence.compare_service import run_compare_and_build_card
from labpilot.research_engine.evidence.metrics_helper import enrich_metrics, fold_stats
from labpilot.research_engine.evidence.models import EvidenceDecision
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.graph.query import query_techniques
from labpilot.research_engine.intelligence.graph.writer import write_graph_edges_from_card
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore


def test_fold_stats_and_enrich_metrics() -> None:
    stats = fold_stats([0.8, 0.9, 0.85])
    assert "cv_mean" in stats and "cv_std" in stats
    enriched = enrich_metrics(
        {"cv_accuracy": 0.85},
        fold_scores=[0.8, 0.9],
        train_time_s=12.5,
        inference_time_s=0.2,
    )
    assert enriched["cv_fold_scores"] == [0.8, 0.9]
    assert enriched["train_time_s"] == 12.5


def test_attribute_techniques_splits_gain() -> None:
    single = attribute_techniques(["Mixup"], cv_gain=0.01, belief_priors={})
    assert single == {"Mixup": 0.01}
    multi = attribute_techniques(
        ["SpecAugment", "EMA"],
        cv_gain=0.01,
        belief_priors={"SpecAugment": 0.9, "EMA": 0.3},
    )
    assert abs(sum(multi.values()) - 0.01) < 1e-9
    assert multi["SpecAugment"] > multi["EMA"]


def test_evidence_card_round_trip(tmp_path: Path) -> None:
    competition = "demo-ev"
    Hypotheses = HypothesisStore(tmp_path, competition)
    hyp = Hypotheses.create(
        observation="try mixup",
        reason="r",
        prediction="p",
        confidence=0.7,
        expected_impact=0.004,
        tags=["Mixup", "stacked", "fork:H-001"],
        technique="Mixup",
        parent_hypothesis_id="H-001",
        technique_stack=["Mixup"],
    )
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition=competition,
        treatment_execution_id="E-014",
        treatment_metrics={
            "cv_accuracy": 0.91,
            "cv_std": 0.01,
            "train_time_s": 100.0,
        },
        control_execution_id="E-008",
        control_metrics={
            "cv_accuracy": 0.90,
            "cv_std": 0.02,
            "train_time_s": 80.0,
        },
        hypothesis_id=hyp.id,
        plan_id="P-002",
        plan_metadata={"change_category": "augmentation", "tags": ["audio"]},
        persist=True,
    )
    assert card.id.startswith("EV-")
    assert card.observed.cv_gain == pytest_approx(0.01)
    assert card.decision == EvidenceDecision.ACCEPTED
    assert card.technique_attribution.get("Mixup") == pytest_approx(0.01)
    assert card.impact_error == pytest_approx(0.006)
    assert "audio" in card.reusable_for or "augmentation" in card.reusable_for

    loaded = EvidenceCardStore(tmp_path, competition).get(card.id)
    assert loaded is not None
    assert loaded.decision == EvidenceDecision.ACCEPTED

    ws = tmp_path / "ws"
    ws.mkdir()
    write_comparison_files(ws, card)
    cmp = json.loads((ws / "comparison.json").read_text())
    assert cmp["cv_delta"] == pytest_approx(0.01)
    assert (ws / "artifacts" / "comparison.json").is_file()


def pytest_approx(val: float, rel: float = 1e-6):
    class _A:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(float(other) - val) <= rel

        def __repr__(self) -> str:
            return f"approx({val})"

    return _A()


def test_missing_control_inconclusive(tmp_path: Path) -> None:
    card = build_evidence_card(
        knowledge_dir=tmp_path,
        competition="demo-miss",
        treatment_execution_id="E-1",
        treatment_metrics={"cv_accuracy": 0.9},
        persist=True,
    )
    assert card.decision == EvidenceDecision.INCONCLUSIVE
    assert "missing_control" in card.decision_reason


def test_compare_capability_builds_card(tmp_path: Path) -> None:
    competition = "cmp-demo"
    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, competition).ensure()
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "metrics.json").write_text(
        json.dumps(
            {
                "cv_accuracy": 0.92,
                "cv_std": 0.01,
                "train_time_s": 50.0,
            }
        ),
        encoding="utf-8",
    )
    Hypotheses = HypothesisStore(knowledge, competition)
    parent = Hypotheses.create(
        observation="parent",
        reason="r",
        prediction="p",
        confidence=0.6,
        tags=["Alpha"],
        technique="Alpha",
    )
    with KnowledgeStore(knowledge, competition) as store:
        from labpilot.research_engine.intelligence.models import (
            ResearchArtifact,
            ResearchArtifactType,
        )

        store.upsert_artifact(
            ResearchArtifact(
                id="exp:execution:E-parent",
                type=ResearchArtifactType.EXPERIMENT,
                source="labpilot",
                title="parent",
                techniques=["Alpha"],
                confidence=0.6,
                competition_slug=competition,
                metadata={
                    "execution_id": "E-parent",
                    "hypothesis_id": parent.id,
                    "metrics": {"cv_accuracy": 0.90, "cv_std": 0.02, "train_time_s": 40.0},
                },
            )
        )

    child = Hypotheses.create(
        observation="child",
        reason="r",
        prediction="p",
        confidence=0.7,
        expected_impact=0.005,
        tags=["Mixup", "fork:" + parent.id],
        technique="Mixup",
        parent_hypothesis_id=parent.id,
    )
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-010",
        competition=competition,
        hypothesis_id=child.id,
        goal="improve",
        status=PlanStatus.READY,
        metadata={
            "parent_hypothesis_id": parent.id,
            "parent_execution_id": "E-parent",
            "parent_metrics": {
                "cv_accuracy": 0.90,
                "cv_std": 0.02,
                "train_time_s": 40.0,
            },
        },
        tasks=[
            ResearchTask(
                id="P-010-T01",
                plan_id="P-010",
                type=TaskType.COMPARE,
                description="compare",
            )
        ],
        created_at=now,
        updated_at=now,
    )
    execution = ResearchExecution(
        id="E-child",
        plan_id=plan.id,
        competition=competition,
    )
    ctx = TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=execution,
        paths=paths,
        workspace_root=ws,
        competition=competition,
    )
    card = run_compare_and_build_card(ctx)
    assert card.observed.cv_gain == pytest_approx(0.02)
    assert card.decision == EvidenceDecision.ACCEPTED
    assert (ws / "comparison.json").is_file()
    cmp = json.loads((ws / "comparison.json").read_text())
    assert cmp["cv_delta"] == pytest_approx(0.02)

    edges = write_graph_edges_from_card(
        knowledge_dir=knowledge, competition=competition, card=card
    )
    assert edges["links"] >= 1
    hits = query_techniques(
        knowledge_dir=knowledge,
        competition=competition,
        min_cv_gain=0.001,
        limit=10,
    )
    assert any(h["technique"] == "Mixup" for h in hits)
