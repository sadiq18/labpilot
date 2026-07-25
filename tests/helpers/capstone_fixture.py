"""Offline seeded fixture for Milestone 3 Plan 11 success-criteria tests.

No network. Micro Agents stay off (rule_engine) unless a test opts in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.providers.capability import CapabilityResult
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
    TransferOpportunity,
)

CAPSTONE_SLUG = "birdclef-2026"


def capstone_artifacts() -> list[ResearchArtifact]:
    """Deterministic Layer-2 artifacts covering README §1 Q1–Q5 evidence."""
    profile = {
        "slug": CAPSTONE_SLUG,
        "title": "BirdCLEF 2026",
        "url": f"https://www.kaggle.com/competitions/{CAPSTONE_SLUG}",
        "problem_type": "Audio Classification",
        "metric": {"name": "macro_f1", "maximize": True},
        "metadata": {"domain": "bioacoustics", "task": "Audio Classification"},
        "winning_solutions": CapabilityResult(
            available=False,
            status="unavailable",
            reason="Not available through configured provider.",
        ).model_dump(mode="json"),
        "related_competitions": [
            {
                "slug": "birdclef-2025",
                "title": "BirdCLEF 2025",
                "relation": "previous_edition",
                "score": 0.9,
            }
        ],
        "capability_notes": [],
        "page_enrichment_source": "rule_engine",
    }
    return [
        ResearchArtifact(
            id=f"competition:{CAPSTONE_SLUG}:profile",
            type=ResearchArtifactType.COMPETITION,
            source="kaggle",
            title="BirdCLEF 2026",
            summary="Imbalanced bioacoustics classification (Macro F1).",
            competition_slug=CAPSTONE_SLUG,
            metadata={"kind": "profile", "profile": profile},
            confidence=0.9,
        ),
        ResearchArtifact(
            id="competition:birdclef-2025",
            type=ResearchArtifactType.COMPETITION,
            source="kaggle",
            title="BirdCLEF 2025",
            competition_slug=CAPSTONE_SLUG,
            metadata={
                "kind": "related",
                "related_slug": "birdclef-2025",
                "relation": "previous_edition",
                "score": 0.9,
                "rationale": "Previous edition",
                "tags_overlap": ["audio", "birds"],
            },
            confidence=0.8,
        ),
        ResearchArtifact(
            id="paper:mixup-macro-f1",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="Mixup improves Macro F1 on imbalanced audio",
            techniques=["Mixup"],
            summary="Mixup consistently improves Macro F1 on imbalanced audio tasks.",
            confidence=0.85,
            competition_slug=CAPSTONE_SLUG,
        ),
        ResearchArtifact(
            id="paper:ema-birdclef",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="EMA for BirdCLEF fine-tuning",
            techniques=["EMA"],
            summary="EMA stabilizes training; no official winning-solution dump.",
            confidence=0.7,
            competition_slug=CAPSTONE_SLUG,
        ),
        ResearchArtifact(
            id="paper:focal-rare",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="Focal Loss for rare classes",
            techniques=["Focal Loss"],
            summary="Focal Loss can help rare species heads when tuned carefully.",
            confidence=0.75,
            competition_slug=CAPSTONE_SLUG,
        ),
        ResearchArtifact(
            id="exp:12",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="Exp 12",
            techniques=["Mixup"],
            summary="Mixup improved macro F1 on BirdCLEF fold (+0.03).",
            confidence=0.92,
            competition_slug=CAPSTONE_SLUG,
        ),
        ResearchArtifact(
            id="exp:14",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="Exp 14",
            techniques=["Mixup"],
            summary="Mixup hurt rare classes when alpha was too high.",
            confidence=0.88,
            competition_slug=CAPSTONE_SLUG,
            metadata={"effect": "hurts"},
        ),
        ResearchArtifact(
            id="exp:19",
            type=ResearchArtifactType.EXPERIMENT,
            source="m2",
            title="Exp 19",
            techniques=["Focal Loss"],
            summary="Focal Loss hurt performance on the rare-species head (delta < 0).",
            confidence=0.9,
            competition_slug=CAPSTONE_SLUG,
            metadata={"effect": "hurts", "metric_delta": -0.02},
        ),
        ResearchArtifact(
            id="repo:owner/audio-pipeline",
            type=ResearchArtifactType.REPOSITORY,
            source="github",
            title="owner/audio-pipeline",
            techniques=["Focal Loss", "SpecAugment"],
            summary="Compatible PyTorch training stack with Focal Loss head.",
            confidence=0.7,
            competition_slug=CAPSTONE_SLUG,
        ),
        ResearchArtifact(
            id="paper:specaugment",
            type=ResearchArtifactType.PAPER,
            source="semantic_scholar",
            title="SpecAugment for ASR",
            techniques=["SpecAugment"],
            summary="SpecAugment improves generalization on audio.",
            confidence=0.8,
            competition_slug=CAPSTONE_SLUG,
        ),
    ]


def capstone_transfers() -> list[dict[str, Any]]:
    return [
        TransferOpportunity(
            repo_id="github:owner/audio-pipeline",
            summary="Focal Loss vs your CE",
            local_baseline="CrossEntropy",
            remote_choice="Focal Loss",
            effort=EffortEstimate.MINUTES_20,
            expected_gain=ExpectedGain.MEDIUM,
            hypothesis_hint="Swap CE → Focal Loss on rare species head",
            deltas=["loss:ce→focal"],
            interesting_files=["train.py"],
        ).model_dump(mode="json")
    ]


class CapstoneAnalyzer:
    """Offline fake analyzer — upserts fixture artifacts; no network."""

    default_enabled = True

    def __init__(self, name: str, items: list[ResearchArtifact], *, transfers=None) -> None:
        self.name = name
        self._items = items
        self._transfers = transfers or []

    def analyze(self, context):
        with KnowledgeStore(context.knowledge_dir, context.competition) as store:
            for artifact in self._items:
                store.upsert_artifact(artifact)
        return ResearchArtifacts(
            analyzer=self.name,
            items=self._items,
            transfers=self._transfers,
        )


def seed_capstone_store(store: KnowledgeStore) -> list[ResearchArtifact]:
    """Seed knowledge.db directly (retrieve / hypothesize paths)."""
    artifacts = capstone_artifacts()
    for artifact in artifacts:
        store.upsert_artifact(artifact)
    KnowledgeHub(store).ingest(artifacts)
    # Promote Mixup after local corroboration — hub never writes validated.
    from labpilot.research_engine.intelligence.knowledge.store import entity_id

    unit_id = entity_id("technique", "Mixup")
    store.upsert_belief(
        belief_id=f"belief_{unit_id}",
        technique="Mixup",
        status="validated",
        effect="improves",
        confidence=0.9,
        metadata={
            "knowledge_unit_id": unit_id,
            "local_evidence": ["exp:12"],
            "external_evidence": ["paper:mixup-macro-f1"],
        },
    )
    store.merge_technique(
        "Focal Loss",
        known_issues="Hurt rare-species head in exp:19",
        confidence=0.7,
    )
    return artifacts


def build_capstone_registry() -> AnalyzerRegistry:
    artifacts = capstone_artifacts()
    profile = [a for a in artifacts if a.metadata.get("kind") == "profile"]
    related = [a for a in artifacts if a.metadata.get("kind") == "related"]
    papers = [a for a in artifacts if a.type is ResearchArtifactType.PAPER]
    experiments = [a for a in artifacts if a.type is ResearchArtifactType.EXPERIMENT]
    repos = [a for a in artifacts if a.type is ResearchArtifactType.REPOSITORY]
    registry = AnalyzerRegistry()
    registry.register(CapstoneAnalyzer("competition", profile + related))
    registry.register(CapstoneAnalyzer("papers", papers))
    registry.register(CapstoneAnalyzer("experiments", experiments))
    registry.register(
        CapstoneAnalyzer("repositories", repos, transfers=capstone_transfers())
    )
    return registry


def run_capstone_analyze(tmp_path: Path) -> tuple[AnalysisReport, Path]:
    """Full offline analyze → validated analyze.json (Micro Agents off)."""
    knowledge_dir = tmp_path / "knowledge"
    runs_dir = tmp_path / "runs"
    context = build_context(
        CAPSTONE_SLUG,
        runs_dir=runs_dir,
        knowledge_dir=knowledge_dir,
    )
    report = AnalyzeOrchestrator(
        build_capstone_registry(),
        llm_client=None,
        ingest_knowledge=True,
        hypothesize=True,
    ).analyze(context)
    # Promote Mixup the way improve would after local corroboration.
    from labpilot.research_engine.intelligence.knowledge.store import entity_id

    unit_id = entity_id("technique", "Mixup")
    with KnowledgeStore(knowledge_dir, CAPSTONE_SLUG) as store:
        store.upsert_belief(
            belief_id=f"belief_{unit_id}",
            technique="Mixup",
            status="validated",
            effect="improves",
            confidence=0.9,
            metadata={
                "knowledge_unit_id": unit_id,
                "local_evidence": ["exp:12"],
                "external_evidence": ["paper:mixup-macro-f1"],
            },
        )
        AnalyzeOrchestrator._refresh_technique_buckets(report, store)
        store.merge_technique(
            "Focal Loss",
            known_issues="Hurt rare-species head in exp:19",
            confidence=0.7,
        )
    from labpilot.research_engine.intelligence.renderers.json import write_report

    path = write_report(report, context.report_path)
    return report, path
