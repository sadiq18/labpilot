"""Plan 9 — Retrieval + Context Builder (no network, no analyze wiring)."""

from __future__ import annotations

import json
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli import main as cli_main
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.retrieval import (
    TOTAL_CHAR_BUDGET,
    ContextBuilder,
    QueryType,
    classify_intent_rules,
    plan_for,
)
from labpilot.research_engine.intelligence.retrieval.compress import compress_bundle
from labpilot.research_engine.intelligence.retrieval.fetchers import SymbolicFetcher
from labpilot.research_engine.intelligence.retrieval.models import (
    RetrievalIntent,
    SymbolicBundle,
)

runner = cli_runner()


def _seed(store: KnowledgeStore) -> None:
    artifacts = [
        ResearchArtifact(
            id="paper:1",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="SpecAugment for ASR",
            techniques=["SpecAugment", "Mixup"],
            summary="SpecAugment improves generalization on audio.",
            confidence=0.8,
        ),
        ResearchArtifact(
            id="paper:2",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="Mixup Classification",
            techniques=["Mixup"],
            summary="Mixup helps macro F1.",
            confidence=0.7,
        ),
        ResearchArtifact(
            id="exp:12",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="exp-12 mixup",
            techniques=["Mixup"],
            summary="Mixup improved macro F1 on BirdCLEF fold.",
            confidence=0.9,
        ),
        ResearchArtifact(
            id="exp:14",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="exp-14 wide mask",
            techniques=["SpecAugment"],
            summary="Large masking decreased recall on rare classes.",
            confidence=0.85,
        ),
        ResearchArtifact(
            id="repo:owner/audio-aug",
            type=ResearchArtifactType.REPOSITORY,
            source="github",
            title="owner/audio-aug",
            techniques=["SpecAugment"],
            summary="PyTorch SpecAugment implementation.",
            confidence=0.6,
        ),
    ]
    for artifact in artifacts:
        store.upsert_artifact(artifact)
    KnowledgeHub(store).ingest(artifacts)
    # Attach a known issue for failure retrieval.
    store.merge_technique(
        "SpecAugment",
        known_issues="Heavy masking hurts small datasets",
        confidence=0.8,
    )


# --- intent -----------------------------------------------------------------


def test_intent_rules_from_competition_profile() -> None:
    intent = classify_intent_rules(
        question="How can I improve BirdCLEF macro F1?",
        profile={
            "slug": "birdclef-2026",
            "task": "Audio Classification",
            "domain": "bioacoustics",
            "metric": {"name": "macro_f1"},
        },
        pipeline=["EMA", "Mixup"],
    )
    assert intent.query_type is QueryType.HYPOTHESIS_GENERATION
    assert intent.domain == "bioacoustics"
    assert intent.metric == "macro_f1"
    assert intent.current_pipeline == ["EMA", "Mixup"]
    assert intent.classified_by == "rules"
    assert "Improve" in (intent.goal or "")


def test_intent_rules_structured_query_keywords() -> None:
    intent = classify_intent_rules(
        question="Find techniques that improve Macro F1 on Audio"
    )
    assert intent.query_type is QueryType.STRUCTURED_QUERY
    assert intent.metric == "macro_f1"


def test_intent_classifier_agent_rule_engine() -> None:
    from labpilot.accessor.common.micro_agents import StructuredContext
    from labpilot.research_engine.intelligence.micro_agents.intent_classifier import (
        IntentClassifierAgent,
    )

    result = IntentClassifierAgent().run(
        StructuredContext(
            text="Explain why Mixup helps",
            data={"profile": {"domain": "bioacoustics"}, "pipeline": ["Mixup"]},
        )
    )
    assert isinstance(result, RetrievalIntent)
    assert result.query_type is QueryType.EXPLAIN


# --- symbolic + expansion ---------------------------------------------------


def test_symbolic_fetches_techniques_and_expands_evidence(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        _seed(store)
        intent = RetrievalIntent(
            query_type=QueryType.HYPOTHESIS_GENERATION,
            domain="bioacoustics",
            need_papers=True,
            need_experiments=True,
            need_repositories=True,
            current_pipeline=["Mixup"],
            question="improve BirdCLEF",
        )
        bundle = SymbolicFetcher(store).fetch(intent, plan_for(intent.query_type))

    names = {row["name"] for row in bundle.techniques}
    assert "SpecAugment" in names
    assert "Mixup" in names
    assert any(hit.document_id == "paper:1" for hit in bundle.papers)
    assert any(hit.document_id == "exp:12" for hit in bundle.experiments)
    assert any(hit.document_id == "repo:owner/audio-aug" for hit in bundle.repositories)
    assert any(hit.kind == "failure" for hit in bundle.failures)
    assert bundle.discussions == []
    assert any("Plan F" in note for note in bundle.notes)


def test_symbolic_soft_empty_when_store_empty(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "empty-comp") as store:
        bundle = SymbolicFetcher(store).fetch(
            RetrievalIntent(query_type=QueryType.STRUCTURED_QUERY),
            plan_for(QueryType.STRUCTURED_QUERY),
        )
    assert bundle.techniques == []
    assert any("no techniques" in note for note in bundle.notes)


# --- compression / budgets --------------------------------------------------


def test_compression_respects_char_budget_and_excludes_raw() -> None:
    from labpilot.research_engine.intelligence.retrieval.models import RetrievalHit

    bundle = SymbolicBundle(
        techniques=[
            {
                "id": "tech_specaugment",
                "name": "SpecAugment",
                "confidence": 0.9,
                "summary": "Improves generalization",
                "known_issues": "Heavy masking hurts small datasets",
                "metadata": "{}",
            }
        ],
        papers=[
            RetrievalHit(
                kind="paper",
                document_id=f"paper:{i}",
                label=f"Paper {i} " + ("title " * 40),
                knowledge_ids=["tech_specaugment"],
                summary="body " * 200,
            )
            for i in range(30)
        ],
        repositories=[
            RetrievalHit(
                kind="repository",
                document_id=f"repo:{i}",
                label=f"repo/{i}",
                knowledge_ids=["tech_specaugment"],
                summary="readme " * 100,
            )
            for i in range(20)
        ],
        experiments=[],
        failures=[],
    )

    intent = RetrievalIntent(
        query_type=QueryType.HYPOTHESIS_GENERATION,
        question="Suggest next experiments",
        current_pipeline=["EMA"],
    )
    cards, fields, brief, budget = compress_bundle(
        bundle, intent=intent, competition={"slug": "birdclef-2026"}
    )
    assert cards
    assert "SpecAugment" in brief
    assert "Current Competition" in brief
    assert "%PDF" not in brief
    assert "full text:" not in brief.lower()
    assert budget["total_chars"] <= TOTAL_CHAR_BUDGET
    assert "techniques" in fields


# --- context builder --------------------------------------------------------


def test_context_builder_returns_typed_research_context(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        _seed(store)
        ctx = ContextBuilder(store).build(
            "How can I improve BirdCLEF?",
            pipeline=["EMA", "Mixup"],
            competition={"slug": "birdclef-2026"},
        )

    assert ctx.intent is not None
    assert ctx.intent.query_type is QueryType.HYPOTHESIS_GENERATION
    assert ctx.techniques
    assert ctx.papers
    assert ctx.experiments
    assert ctx.brief
    assert "Technique:" in ctx.brief
    assert ctx.budget["total_chars"] <= TOTAL_CHAR_BUDGET
    # Round-trip serializable for Plan 10.
    payload = json.loads(ctx.model_dump_json())
    assert "brief" in payload and "techniques" in payload


def test_context_builder_accepts_prebuilt_intent(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        _seed(store)
        intent = RetrievalIntent(
            query_type=QueryType.STRUCTURED_QUERY,
            question="techniques for audio",
            need_repositories=False,
        )
        ctx = ContextBuilder(store).build(intent)
    assert ctx.intent is not None
    assert ctx.intent.query_type is QueryType.STRUCTURED_QUERY
    assert ctx.repositories == []


# --- CLI --------------------------------------------------------------------


def test_retrieve_cli_text_and_json(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        _seed(store)

    text = runner.invoke(
        cli_main.app,
        [
            "retrieve",
            "birdclef-2026",
            "--question",
            "How can I improve BirdCLEF?",
            "--pipeline",
            "EMA,Mixup",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert text.exit_code == 0, text.stdout
    assert "Research retrieval" in text.stdout
    assert "SpecAugment" in text.stdout or "Mixup" in text.stdout

    raw = runner.invoke(
        cli_main.app,
        [
            "retrieve",
            "birdclef-2026",
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert raw.exit_code == 0, raw.stdout
    payload = json.loads(raw.stdout)
    assert "brief" in payload
    assert "techniques" in payload


def test_retrieve_cli_fails_when_empty(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_main.app,
        [
            "retrieve",
            "missing-comp",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 1
    assert "No knowledge found" in result.stdout
