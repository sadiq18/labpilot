"""Context Engine ExperienceProvider (cross-competition transfer)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.context import (
    ContextRequest,
    ExperienceProvider,
    build_context,
)
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.models import ExperienceFacet
from labpilot.research_engine.memory.seed import write_seed_manifest
from labpilot.workspace import scaffold_workspace


def _seed_store(knowledge: Path) -> None:
    store = ExperienceStore(knowledge)
    try:
        store.create(
            source_competition="birdclef-2026",
            idempotency_key="bird-1",
            goal="Improve BirdCLEF score",
            hypothesis="SpecAugment helps minority classes",
            action="Added SpecAugment + EMA",
            result="+0.006 LB",
            outcome="success",
            facets=[
                ExperienceFacet(
                    facet="audio",
                    confidence=0.82,
                    evidence=["bird", "spectrogram"],
                    source="rules",
                ),
                ExperienceFacet(
                    facet="augmentation",
                    confidence=0.65,
                    evidence=["specaugment"],
                    source="rules",
                ),
            ],
        )
        store.create(
            source_competition="titanic",
            idempotency_key="tab-1",
            goal="Survive classification",
            action="Dropped columns",
            result="worse",
            outcome="fail",
            facets=[
                ExperienceFacet(
                    facet="tabular",
                    confidence=0.7,
                    evidence=["table"],
                    source="rules",
                )
            ],
        )
    finally:
        store.close()


def test_experience_provider_surfaces_cross_comp(tmp_path: Path) -> None:
    research = tmp_path / "kaggle"
    bird = scaffold_workspace(research / "birdclef-2026", "birdclef-2026")
    whale = scaffold_workspace(research / "whale-sound", "whale-sound")
    # Shared experiences.db via parent research root
    _seed_store(bird.knowledge_dir)

    request = ContextRequest(
        competition="whale-sound",
        goal="Underwater sound classification",
        query="audio spectrogram augmentation bird",
        knowledge_dir=whale.knowledge_dir,
        max_items=16,
        max_chars=8000,
    )
    bundle = build_context(request, providers=[ExperienceProvider()])
    exp_items = [i for i in bundle.items if i.source == "experience"]
    assert exp_items, bundle.provider_errors
    # Cross-comp: birdclef record must survive competition filters
    sources = {i.metadata.get("source_competition") for i in exp_items}
    assert "birdclef-2026" in sources
    assert all("competition" not in i.metadata for i in exp_items)
    bird_hit = next(
        i for i in exp_items if i.metadata.get("source_competition") == "birdclef-2026"
    )
    assert "SpecAugment" in bird_hit.text or "audio" in bird_hit.text.lower()


def test_seeded_experiences_get_boost(tmp_path: Path) -> None:
    research = tmp_path / "kaggle"
    bird = scaffold_workspace(research / "bird", "bird")
    whale = scaffold_workspace(research / "whale", "whale")
    _seed_store(bird.knowledge_dir)
    store = ExperienceStore(bird.knowledge_dir)
    try:
        records = store.list(source_competition="birdclef-2026")
    finally:
        store.close()
    write_seed_manifest(
        whale.knowledge_dir,
        target_competition="whale",
        source_competition="birdclef-2026",
        records=records,
    )

    request = ContextRequest(
        competition="whale",
        query="audio",
        knowledge_dir=whale.knowledge_dir,
        max_items=16,
    )
    items = ExperienceProvider()._fetch_sync(request)
    bird_item = next(
        i for i in items if i.metadata.get("source_competition") == "birdclef-2026"
    )
    assert bird_item.metadata.get("seeded") is True
    assert "(seeded)" in bird_item.reason
    assert bird_item.score >= 0.7
