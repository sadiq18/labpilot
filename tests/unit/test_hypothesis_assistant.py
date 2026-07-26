"""Plan 10 — Hypothesis Assistant (recommendations only; no network / no execute)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli import main as cli_main
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import HypothesisCreatedBy, HypothesisStatus
from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.hypothesis import (
    HypothesisAssistant,
    generate_candidates,
    rank_candidates,
    score_candidate,
)
from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisCandidate,
    HypothesisCandidateKind,
)
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
    TransferOpportunity,
)
from labpilot.research_engine.intelligence.retrieval import ContextBuilder
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryType,
    ResearchContext,
    RetrievalIntent,
)

runner = CliRunner()


def _seed(store: KnowledgeStore) -> None:
    artifacts = [
        ResearchArtifact(
            id="paper:1",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="SpecAugment for ASR",
            techniques=["SpecAugment", "Mixup"],
            summary="SpecAugment improves generalization.",
            confidence=0.8,
        ),
        ResearchArtifact(
            id="exp:14",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="exp-14 wide mask",
            techniques=["SpecAugment"],
            summary="Large masking decreased recall.",
            confidence=0.85,
        ),
        ResearchArtifact(
            id="repo:owner/audio",
            type=ResearchArtifactType.REPOSITORY,
            source="github",
            title="owner/audio",
            techniques=["Focal Loss"],
            summary="Rare-class focal loss head.",
            confidence=0.7,
        ),
    ]
    for artifact in artifacts:
        store.upsert_artifact(artifact)
    KnowledgeHub(store).ingest(artifacts)
    store.merge_technique(
        "SpecAugment",
        known_issues="Heavy masking hurts small datasets",
        confidence=0.8,
    )


class FakeAnalyzer:
    default_enabled = True

    def __init__(self, name: str, items: list[ResearchArtifact], transfers=None) -> None:
        self.name = name
        self._items = items
        self._transfers = transfers or []

    def analyze(self, context):
        with KnowledgeStore(context.knowledge_dir, context.competition) as store:
            for artifact in self._items:
                store.upsert_artifact(artifact)
        return ResearchArtifacts(
            analyzer=self.name, items=self._items, transfers=self._transfers
        )


def test_score_formula_orders_high_gain_low_effort_first() -> None:
    high = HypothesisCandidate(
        key="a",
        kind=HypothesisCandidateKind.PIPELINE_DIFF,
        title="Add Focal Loss",
        expected_impact=ExpectedGain.HIGH,
        confidence=0.9,
        implementation_effort=EffortEstimate.MINUTES_20,
    )
    low = HypothesisCandidate(
        key="b",
        kind=HypothesisCandidateKind.TECHNIQUE,
        title="Rebuild architecture",
        expected_impact=ExpectedGain.LOW,
        confidence=0.4,
        implementation_effort=EffortEstimate.DAYS,
    )
    assert score_candidate(high) > score_candidate(low)
    ranked = rank_candidates([low, high], limit=2)
    assert ranked[0][1].key == "a"


def test_generate_candidates_includes_pipeline_diff_and_failure_fix() -> None:
    context = ResearchContext(
        techniques=[
            {
                "id": "tech_specaugment",
                "name": "SpecAugment",
                "confidence": 0.8,
                "paper_ids": ["paper:1"],
                "experiment_ids": ["exp:14"],
                "repository_ids": [],
                "known_issues": "Heavy masking hurts small datasets",
            },
            {
                "id": "tech_mixup",
                "name": "Mixup",
                "confidence": 0.7,
                "paper_ids": ["paper:1"],
                "experiment_ids": [],
                "repository_ids": [],
            },
        ],
        failures=[
            {
                "kind": "failure",
                "document_id": "exp:14",
                "label": "known issue: SpecAugment",
                "summary": "Large masking decreased recall",
            }
        ],
        intent=RetrievalIntent(
            query_type=QueryType.HYPOTHESIS_GENERATION,
            current_pipeline=["Mixup"],
        ),
    )
    transfers = [
        TransferOpportunity(
            repo_id="github:owner/audio",
            summary="Swap CE for Focal Loss",
            remote_choice="Focal Loss",
            effort=EffortEstimate.MINUTES_20,
            expected_gain=ExpectedGain.MEDIUM,
            hypothesis_hint="Use Focal Loss on rare classes",
        )
    ]
    candidates = generate_candidates(context, transfers=transfers)
    kinds = {c.kind for c in candidates}
    assert HypothesisCandidateKind.PIPELINE_DIFF in kinds
    assert HypothesisCandidateKind.FAILURE_FIX in kinds
    assert HypothesisCandidateKind.TRANSFER in kinds
    # Mixup already in pipeline → pipeline_diff for Mixup should be absent.
    assert not any(
        c.kind is HypothesisCandidateKind.PIPELINE_DIFF and c.technique == "Mixup"
        for c in candidates
    )


def test_assistant_emits_at_most_10_and_persists_suggested(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)

    result = HypothesisAssistant(created_by=HypothesisCreatedBy.HYPOTHESIZE).recommend(
        knowledge_dir=knowledge_dir,
        competition="birdclef-2026",
        question="Suggest next experiments",
        pipeline=["EMA"],
        limit=10,
        persist=True,
        write_report=True,
        progressive=True,
    )
    assert 1 <= len(result.recommendations) <= 10
    for card in result.recommendations:
        assert card.hypothesis_id.startswith("H-")
        assert card.title
        assert card.supporting_evidence or card.reason
        assert card.created_by is HypothesisCreatedBy.HYPOTHESIZE
        assert card.generator is not None
        assert str(card.origin)

    store = HypothesisStore(knowledge_dir, "birdclef-2026")
    hyps = store.list(status=HypothesisStatus.PROPOSED)
    assert len(hyps) == len(result.recommendations)
    assert all(h.created_by is not None for h in hyps)
    assert all(h.source == "analyze" for h in hyps)

    with KnowledgeStore(knowledge_dir, "birdclef-2026") as kstore:
        db_rows = kstore.list_hypotheses(status="proposed")
        assert {row["id"] for row in db_rows} == {h.id for h in hyps}
        # Every generated hypothesis must carry a non-zero impact estimate.
        assert all(row["expected_impact"] > 0.0 for row in db_rows)
    assert all(h.expected_impact > 0.0 for h in hyps)
    assert all(card.expected_impact_value > 0.0 for card in result.recommendations)

    report_path = knowledge_dir / "birdclef-2026/research/reports/hypotheses.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text())
    assert len(payload["recommendations"]) == len(result.recommendations)


def test_assistant_does_not_execute_training(tmp_path: Path, monkeypatch) -> None:
    """Guard: assistant must never call Engineer / run entry points."""
    calls: list[str] = []

    def boom(*_a, **_k):
        calls.append("engineer")
        raise AssertionError("must not execute")

    monkeypatch.setattr(
        "labpilot.research_engine.execution.engineer.ResearchEngineer.run_plan",
        boom,
        raising=False,
    )
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)
    HypothesisAssistant().recommend(
        knowledge_dir=knowledge_dir,
        competition="birdclef-2026",
        pipeline=["EMA"],
        persist=False,
    )
    assert calls == []


def test_progressive_context_builder_notes_passes(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        _seed(store)
        ctx = ContextBuilder(store).build(
            "Suggest next experiments",
            pipeline=["EMA"],
            progressive=True,
            core_technique_limit=4,
        )
    assert any("progressive: pass1_core=" in note for note in ctx.notes)
    assert ctx.techniques
    assert ctx.brief


def test_orchestrator_wires_hypothesis_recommendations(tmp_path: Path) -> None:
    ctx = build_context(
        "birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )
    artifacts = [
        ResearchArtifact(
            id="paper:1",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="P1",
            techniques=["SpecAugment", "Mixup"],
            confidence=0.8,
        )
    ]
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", artifacts))
    report = AnalyzeOrchestrator(reg, hypothesize=True).analyze(ctx)
    assert report.hypothesis_recommendations
    assert report.summary["hypothesis_count"] == len(report.hypothesis_recommendations)
    assert report.retrieval.papers or report.knowledge_units


def test_orchestrator_can_skip_hypothesize(tmp_path: Path) -> None:
    ctx = build_context(
        "birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )
    artifacts = [
        ResearchArtifact(
            id="paper:1",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="P1",
            techniques=["Mixup"],
        )
    ]
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", artifacts))
    report = AnalyzeOrchestrator(reg, hypothesize=False).analyze(ctx)
    assert report.hypothesis_recommendations == []
    assert any("hypothesis] skipped" in note for note in report.notes)


def test_hypothesize_cli(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)

    result = runner.invoke(
        cli_main.app,
        [
            "hypothesize",
            "birdclef-2026",
            "--pipeline",
            "EMA",
            "--limit",
            "5",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Hypothesis Assistant" in result.stdout
    assert "new hypothesis generated" in result.stdout
    assert "#" in result.stdout


def test_hypothesize_new_subcommand_matches_bare_slug(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)

    result = runner.invoke(
        cli_main.app,
        [
            "hypothesize",
            "new",
            "birdclef-2026",
            "--pipeline",
            "EMA",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "new hypothesis generated" in result.stdout


def test_rerun_generates_no_duplicate_hypotheses(tmp_path: Path) -> None:
    """Second pass must not re-mint hypotheses already open in the backlog."""
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)

    def _run():
        return HypothesisAssistant(created_by=HypothesisCreatedBy.HYPOTHESIZE).recommend(
            knowledge_dir=knowledge_dir,
            competition="birdclef-2026",
            pipeline=["EMA"],
            persist=True,
            progressive=True,
        )

    first = _run()
    assert first.new_count == len(first.recommendations) >= 1

    second = _run()
    assert second.new_count == 0
    assert len(HypothesisStore(knowledge_dir, "birdclef-2026").list()) == first.new_count
