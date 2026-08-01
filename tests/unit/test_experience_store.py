"""Unit tests for ExperienceStore."""

from __future__ import annotations

from pathlib import Path

from labpilot.accessor.sqlite import SCHEMA_VERSION, SqliteClient
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.models import ExperienceArtifacts, ExperienceFacet


def test_migrate_adds_experience_records_table(tmp_path: Path) -> None:
    db = tmp_path / "experiences.db"
    client = SqliteClient(db)
    try:
        assert client.schema_version() == SCHEMA_VERSION == "8"
        tables = {
            row["name"]
            for row in client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "experience_records" in tables
    finally:
        client.close()


def test_upsert_idempotent_by_key(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = ExperienceStore(knowledge)
    try:
        first = store.create(
            source_competition="birdclef-2026",
            idempotency_key="run-001",
            goal="Improve BirdCLEF score",
            hypothesis="SpecAugment helps minority classes",
            action="Added SpecAugment + EMA",
            result="+0.006 LB",
            outcome="success",
            artifacts=ExperienceArtifacts(
                experiment_id="run-001",
                git_commit="abc123",
                metrics={"lb_score": 0.706},
            ),
            facets=[
                ExperienceFacet(
                    facet="audio",
                    confidence=0.82,
                    evidence=["bird", "clef"],
                    source="rules",
                ),
                ExperienceFacet(
                    facet="augmentation",
                    confidence=0.65,
                    evidence=["specaugment", "ema"],
                    source="rules",
                ),
            ],
        )
        assert first.id == "XR-001"
        assert first.artifacts.git_commit == "abc123"
        assert first.facets[0].evidence == ["bird", "clef"]

        second = store.create(
            source_competition="birdclef-2026",
            idempotency_key="run-001",
            goal="Improve BirdCLEF score",
            hypothesis="SpecAugment helps minority classes",
            action="Added SpecAugment + EMA (retry)",
            result="+0.007 LB",
            outcome="success",
            artifacts=ExperienceArtifacts(
                experiment_id="run-001",
                git_commit="def456",
            ),
            facets=[
                ExperienceFacet(
                    facet="audio",
                    confidence=0.82,
                    evidence=["bird"],
                    source="rules",
                ),
                ExperienceFacet(
                    facet="augmentation",
                    confidence=0.65,
                    evidence=["specaugment"],
                    source="rules",
                ),
                ExperienceFacet(
                    facet="imbalance",
                    confidence=0.45,
                    evidence=["minority"],
                    source="rules",
                ),
            ],
        )
        assert second.id == first.id == "XR-001"
        assert second.action.endswith("(retry)")
        assert second.artifacts.git_commit == "def456"
        assert store.list() == [second]
        assert store.get_by_idempotency_key("run-001") is not None
    finally:
        store.close()


def test_list_filters_cross_competition(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path / "knowledge")
    try:
        store.create(
            source_competition="birdclef-2026",
            idempotency_key="a-1",
            outcome="success",
            facets=[
                ExperienceFacet(facet="audio", confidence=0.8, evidence=["bird"], source="rules"),
                ExperienceFacet(
                    facet="augmentation", confidence=0.6, evidence=["mixup"], source="rules"
                ),
            ],
            result="ok",
        )
        store.create(
            source_competition="titanic",
            idempotency_key="b-1",
            outcome="fail",
            tags=["tabular"],
            result="worse",
        )
        store.create(
            source_competition="birdclef-2026",
            idempotency_key="a-2",
            outcome="fail",
            facets=[
                ExperienceFacet(facet="audio", confidence=0.5, evidence=["audio"], source="rules")
            ],
            result="nope",
        )

        assert len(store.list()) == 3
        bird = store.list(source_competition="birdclef-2026")
        assert {r.idempotency_key for r in bird} == {"a-1", "a-2"}
        fails = store.list(outcome="fail")
        assert {r.idempotency_key for r in fails} == {"b-1", "a-2"}
        audio_aug = store.list(facets=["audio", "augmentation"])
        assert [r.idempotency_key for r in audio_aug] == ["a-1"]
        by_tag = store.list(tag="tabular")
        assert [r.source_competition for r in by_tag] == ["titanic"]
        # Legacy string tags coerce to facets
        assert by_tag[0].facets[0].source == "legacy"
    finally:
        store.close()
