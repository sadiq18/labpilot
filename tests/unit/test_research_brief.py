"""Research Brief builder + markdown renderer (offline)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.intelligence.brief.builder import build_research_brief
from labpilot.research_engine.intelligence.brief.models import ResearchBrief
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    ResearchArtifact,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.renderers.markdown import (
    render_brief_markdown,
    write_brief,
)


def _seed_report(store: KnowledgeStore) -> AnalysisReport:
    dataset = ResearchArtifact(
        id="dataset:birdclef-2026",
        type=ResearchArtifactType.DATASET,
        source="m2",
        title="birdclef dataset",
        summary="audio dataset — 1000 train rows, 12 columns, target=primary_label",
        competition_slug="birdclef-2026",
        metadata={
            "modality": "audio",
            "row_count": 1000,
            "column_count": 12,
            "target_column": "primary_label",
            "null_heavy_columns": ["latitude"],
            "warnings": ["rare classes under-represented"],
        },
    )
    experiment = ResearchArtifact(
        id="exp:12",
        type=ResearchArtifactType.EXPERIMENT,
        source="m2",
        title="exp-12",
        summary="Focal Loss hurt Macro F1 on rare classes",
        techniques=["Focal Loss"],
        competition_slug="birdclef-2026",
        metadata={"failures": ["Focal Loss regression on rare classes"]},
    )
    discussion = ResearchArtifact(
        id="discussion:1",
        type=ResearchArtifactType.DISCUSSION,
        source="kaggle",
        title="LB shake-up",
        summary="Public LB is noisy",
        competition_slug="birdclef-2026",
        metadata={
            "forum_extract": {
                "mistakes": ["target leakage on folds"],
                "dataset_bugs": ["corrupt wav files"],
                "lb_shakeups": ["public LB overfit"],
            }
        },
    )
    paper = ResearchArtifact(
        id="paper:1",
        type=ResearchArtifactType.PAPER,
        source="semantic_scholar",
        title="SpecAugment for ASR",
        techniques=["SpecAugment"],
        confidence=0.8,
    )
    for artifact in (dataset, experiment, discussion, paper):
        store.upsert_artifact(artifact)
    KnowledgeHub(store).ingest([dataset, experiment, discussion, paper])

    return AnalysisReport(
        competition={
            "slug": "birdclef-2026",
            "title": "BirdCLEF 2026",
            "problem_type": "multilabel audio classification",
            "metric": {"name": "Macro F1", "direction": "maximize"},
            "rules_excerpt": "No external pretrained weights on private data.",
            "winning_solutions": {"status": "unavailable"},
            "external_data": {"allowed": False},
        },
        artifacts=[dataset, experiment, discussion, paper],
        papers=[{"id": "paper:1", "title": "SpecAugment for ASR", "techniques": ["SpecAugment"]}],
        repositories=[
            {"id": "repo:1", "title": "owner/audio", "techniques": ["Mixup"]}
        ],
        related_competitions=[
            {"slug": "birdclef-2025", "title": "BirdCLEF 2025", "relation": "previous_edition"}
        ],
        hypothesis_recommendations=[
            {
                "rank": 1,
                "hypothesis_id": "H-001",
                "title": "Try SpecAugment with mild masking",
            }
        ],
        suggested_experiments=[
            {"rank": 1, "title": "Try SpecAugment with mild masking"}
        ],
        techniques={
            "external_recommendations": ["SpecAugment"],
            "locally_validated": [],
            "unverified": ["Mixup"],
        },
    )


def test_research_brief_has_all_sections(tmp_path: Path) -> None:
    with KnowledgeStore(tmp_path / "knowledge", "birdclef-2026") as store:
        report = _seed_report(store)
        brief = build_research_brief(report, store, llm_client=None)

    assert isinstance(brief, ResearchBrief)
    assert brief.generated_by == "llm"
    assert brief.problem_summary
    assert "audio" in brief.dataset_overview.lower() or "1000" in brief.dataset_overview
    assert "Macro F1" in brief.rules_and_metric
    assert brief.related_papers
    assert brief.similar_competitions
    assert brief.repositories
    assert brief.winning_techniques
    assert brief.beliefs
    assert brief.top_hypotheses
    assert any("leakage" in r.lower() or "corrupt" in r.lower() for r in brief.known_risks)
    assert brief.suggested_experiments

    md = render_brief_markdown(brief)
    for heading in (
        "Problem summary",
        "Dataset overview",
        "Competition rules & metric",
        "Related papers",
        "Similar competitions",
        "Relevant GitHub repositories",
        "Winning techniques",
        "Existing beliefs",
        "Top hypotheses",
        "Known risks",
        "Suggested next experiments",
    ):
        assert f"## {heading}" in md

    path = write_brief(brief, tmp_path / "research_brief.md")
    assert path.is_file()
    assert "Research Brief" in path.read_text()
