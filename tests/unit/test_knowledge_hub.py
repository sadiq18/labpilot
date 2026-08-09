"""Plan 8 — Knowledge Extraction hub: alias merging and the belief status gate."""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.knowledge import (
    BeliefStatus,
    EntityType,
    KnowledgeExtractor,
    KnowledgeHub,
    KnowledgeMerger,
    KnowledgeStore,
    entity_id,
)
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry


def _ctx(tmp_path: Path) -> AnalyzeContext:
    return build_context(
        "birdclef-2026", runs_dir=tmp_path / "runs", knowledge_dir=tmp_path / "knowledge"
    )


def _paper(
    pid: str, techniques: list[str], *, source: str = "semantic_scholar"
) -> ResearchArtifact:
    return ResearchArtifact(
        id=pid,
        type=ResearchArtifactType.PAPER,
        source=source,
        title=pid,
        techniques=techniques,
        confidence=0.6,
    )


def _experiment(eid: str, techniques: list[str]) -> ResearchArtifact:
    return ResearchArtifact(
        id=eid,
        type=ResearchArtifactType.EXPERIMENT,
        source="m2",
        title=eid,
        techniques=techniques,
    )


class FakeAnalyzer:
    default_enabled = True

    def __init__(self, name: str, items: list[ResearchArtifact]) -> None:
        self.name = name
        self._items = items

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        return ResearchArtifacts(analyzer=self.name, items=self._items)


# --- extractor --------------------------------------------------------------


def test_extractor_only_pulls_requested_entity_types() -> None:
    artifact = ResearchArtifact(
        id="paper:1",
        type=ResearchArtifactType.PAPER,
        source="semantic_scholar",
        techniques=["Mixup"],
        models=["EfficientNet"],
    )
    candidates = KnowledgeExtractor().extract([artifact])
    assert [(c.entity_type, c.name) for c in candidates] == [(EntityType.TECHNIQUE, "Mixup")]

    both = KnowledgeExtractor().extract(
        [artifact], entity_types=(EntityType.TECHNIQUE, EntityType.ARCHITECTURE)
    )
    assert (EntityType.ARCHITECTURE, "EfficientNet") in [(c.entity_type, c.name) for c in both]


# --- merger -----------------------------------------------------------------


def test_merger_collapses_five_aliases_into_one_concept() -> None:
    artifacts = [
        _paper("paper:1", ["SpecAugment"]),
        _paper("paper:2", ["spec-augment"]),
        _paper("paper:3", ["Time Masking"]),
        _paper("paper:4", ["Frequency Masking"]),
        _paper("paper:5", ["SpecAug"]),
    ]
    clusters = KnowledgeMerger().merge(KnowledgeExtractor().extract(artifacts))
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.canonical == "SpecAugment"
    assert cluster.normalized_by == "alias_seed"
    assert len(cluster.evidence) == 5
    assert "Time Masking" in cluster.aliases


def test_merger_keeps_distinct_concepts_apart() -> None:
    artifacts = [_paper("paper:1", ["Mixup", "Focal Loss", "EMA"])]
    clusters = KnowledgeMerger().merge(KnowledgeExtractor().extract(artifacts))
    assert {c.canonical for c in clusters} == {"Mixup", "Focal Loss", "EMA"}


def test_merger_picks_most_frequent_variant_without_llm() -> None:
    artifacts = [
        _paper("paper:1", ["Attention Pooling"]),
        _paper("paper:2", ["attention pooling"]),
        _paper("paper:3", ["attention pooling"]),
    ]
    clusters = KnowledgeMerger().merge(KnowledgeExtractor().extract(artifacts))
    assert len(clusters) == 1
    assert clusters[0].canonical == "attention pooling"
    assert clusters[0].aliases == ["Attention Pooling"]


def test_merger_ignores_agent_canonical_outside_cluster() -> None:
    class Result:
        canonical = "Something Unrelated"
        category = "augmentation"

    class Agent:
        last_used_llm = True

        def __init__(self, *_args, **_kwargs) -> None: ...

        def run(self, _context):
            return Result()

    import labpilot.research_engine.intelligence.knowledge.merger as merger_mod

    original = merger_mod.ConceptNormalizerAgent
    merger_mod.ConceptNormalizerAgent = Agent  # type: ignore[assignment]
    try:
        artifacts = [_paper("paper:1", ["Attention Pooling"]), _paper("paper:2", ["attn pooling"])]
        clusters = merger_mod.KnowledgeMerger(llm_client=object()).merge(
            KnowledgeExtractor().extract(artifacts)
        )
    finally:
        merger_mod.ConceptNormalizerAgent = original  # type: ignore[assignment]
    assert clusters[0].canonical in {"Attention Pooling", "attn pooling"}
    assert clusters[0].normalized_by == "rule_engine"


# --- hub --------------------------------------------------------------------


def test_hub_ingest_writes_unit_card_and_suggested_belief(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifacts = [_paper("paper:1", ["Mixup"]), _paper("paper:2", ["mix up"])]
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        for artifact in artifacts:
            store.upsert_artifact(artifact)
        result = KnowledgeHub(store).ingest(artifacts)

        assert len(result.units) == 1
        unit = result.units[0]
        assert unit.id == entity_id("technique", "Mixup")
        assert len(unit.evidence) == 2
        assert unit.external_evidence == ["paper:1", "paper:2"]
        assert unit.local_evidence == []

        # Layer 3 row + join links exist.
        assert store.get_technique(unit.id) is not None
        assert store.artifacts_for_technique(unit.id) == ["paper:1", "paper:2"]

        # Layer 4 belief is Suggested — external reading is never validated here.
        belief = result.beliefs[0]
        assert belief.status is BeliefStatus.SUGGESTED
        assert belief.confidence.local == 0.0
        row = store.get_belief(belief.id)
        assert row is not None
        assert row["status"] == "suggested"
        assert json.loads(row["metadata"])["knowledge_unit_id"] == unit.id

    card = ctx.paths.knowledge_dir / "techniques" / f"{unit.id}.json"
    assert json.loads(card.read_text())["name"] == "Mixup"


def test_hub_marks_locally_evidenced_technique_as_testing(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifacts = [_paper("paper:1", ["Mixup"]), _experiment("exp:7", ["Mixup"])]
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        for artifact in artifacts:
            store.upsert_artifact(artifact)
        result = KnowledgeHub(store).ingest(artifacts)
        belief = result.beliefs[0]
        assert belief.status is BeliefStatus.TESTING
        assert belief.local_evidence == ["exp:7"]
        assert belief.confidence.local > 0.0
        # Never promoted past testing by the hub.
        assert not store.list_beliefs(status="validated")
        assert not store.list_beliefs(status="established")


def test_hub_ingest_is_idempotent(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifacts = [_paper("paper:1", ["Mixup"])]
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        store.upsert_artifact(artifacts[0])
        hub = KnowledgeHub(store)
        hub.ingest(artifacts)
        hub.ingest(artifacts)
        assert len(store.list_beliefs()) == 1
        assert store.artifacts_for_technique(entity_id("technique", "Mixup")) == ["paper:1"]


def test_hub_receipt_detects_new_and_changed_artifacts(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifact = _paper("paper:1", ["Mixup"])
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        store.upsert_artifact(artifact)
        hub = KnowledgeHub(store)
        assert [item.id for item in hub.pending_artifacts([artifact])] == ["paper:1"]

        hub.ingest([artifact])
        assert hub.pending_artifacts([artifact]) == []

        changed = artifact.model_copy(update={"techniques": ["CutMix"]})
        store.upsert_artifact(changed)
        assert [item.id for item in hub.pending_artifacts([changed])] == ["paper:1"]


def test_hub_marks_artifact_without_concepts_ingested(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifact = _paper("paper:1", [])
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        store.upsert_artifact(artifact)
        hub = KnowledgeHub(store)
        result = hub.ingest([artifact])
        assert result.units == []
        assert hub.pending_artifacts([artifact]) == []


def test_hub_skips_unpersisted_artifacts_for_links(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifacts = [_paper("paper:1", ["Mixup"])]  # never upserted
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        result = KnowledgeHub(store).ingest(artifacts)
    unit = result.units[0]
    assert unit.external_evidence == ["paper:1"]
    assert unit.metadata["linked_evidence"] == []


def test_hub_notes_when_no_concepts(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifact = ResearchArtifact(
        id="paper:1", type=ResearchArtifactType.PAPER, source="semantic_scholar"
    )
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        result = KnowledgeHub(store).ingest([artifact])
    assert result.units == []
    assert any("no concept candidates" in note for note in result.notes)


def test_hub_supports_non_technique_entities(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    artifact = ResearchArtifact(
        id="paper:1",
        type=ResearchArtifactType.PAPER,
        source="semantic_scholar",
        models=["EfficientNet-B0"],
    )
    with KnowledgeStore(ctx.knowledge_dir, ctx.competition) as store:
        store.upsert_artifact(artifact)
        result = KnowledgeHub(store).ingest(
            [artifact], entity_types=(EntityType.ARCHITECTURE,), write_beliefs=False
        )
        unit = result.units[0]
        assert unit.entity_type is EntityType.ARCHITECTURE
        assert store.get_entity("architecture", unit.id) is not None
        assert store.list_beliefs() == []
    assert (ctx.paths.knowledge_dir / "architectures" / f"{unit.id}.json").exists()


# --- orchestrator wiring ----------------------------------------------------


def test_orchestrator_ingests_once_and_fills_knowledge_units(tmp_path: Path) -> None:
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", [_paper("paper:1", ["SpecAugment"])]))
    reg.register(FakeAnalyzer("experiments", [_paper("paper:2", ["Time Masking"])]))
    report = AnalyzeOrchestrator(reg).analyze(_ctx(tmp_path))

    assert report.summary["knowledge_unit_count"] == 1
    unit = report.knowledge_units[0]
    assert unit["name"] == "SpecAugment"
    assert len(unit["evidence"]) == 2
    assert report.techniques.external_recommendations == ["SpecAugment"]
    assert any("knowledge hub" in note for note in report.notes)


def test_orchestrator_can_disable_ingest(tmp_path: Path) -> None:
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", [_paper("paper:1", ["Mixup"])]))
    report = AnalyzeOrchestrator(reg, ingest_knowledge=False).analyze(_ctx(tmp_path))
    assert report.knowledge_units == []
    assert report.summary["knowledge_unit_count"] == 0
    assert any("ingestion skipped" in note for note in report.notes)


def test_orchestrator_soft_fails_on_hub_error(tmp_path: Path, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("db locked")

    monkeypatch.setattr("labpilot.research_engine.intelligence.orchestrator.KnowledgeStore", boom)
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", [_paper("paper:1", ["Mixup"])]))
    report = AnalyzeOrchestrator(reg).analyze(_ctx(tmp_path))
    assert report.knowledge_units == []
    assert any("ingest failed" in note for note in report.notes)


def test_a_record_reference_is_dropped_not_persisted():
    """`hyp:H-010` points at a hypothesis; it is not a technique anyone can
    test. `merge_technique` refuses these and its error says to filter first —
    no caller did, so `research ingest` died on the first one it met."""
    from labpilot.research_engine.shared.labels import is_record_reference

    assert is_record_reference("hyp:H-010") is True
    assert is_record_reference("rolling_features") is False
