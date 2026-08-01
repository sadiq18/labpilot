"""Unit tests for ReflectionStore and reflection schema (Milestone 6 Plan 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.accessor.sqlite import SCHEMA_VERSION, SqliteClient
from labpilot.research_engine.reflection.store import ReflectionStore


_REFLECTION_TABLES = {
    "experiment_evidence",
    "belief_updates",
    "lessons",
    "research_claims",
    "claim_evidence",
}


def test_migrate_adds_reflection_tables(tmp_path: Path) -> None:
    db = tmp_path / "knowledge.db"
    client = SqliteClient(db)
    try:
        assert client.schema_version() == SCHEMA_VERSION == "7"
        tables = {
            row["name"]
            for row in client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert _REFLECTION_TABLES <= tables
        assert "research_executions" in tables
        assert "beliefs" in tables
    finally:
        client.close()


def test_evidence_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = ReflectionStore(knowledge, "demo")
    try:
        evidence = store.create_evidence(
            execution_id="E-001",
            plan_id="P-001",
            metrics={"cv_score": 0.82},
            comparison={"delta": 0.01, "baseline": 0.81},
            strength="strong",
        )
        assert evidence["id"] == "EE-001"
        assert evidence["competition_slug"] == "demo"
        assert evidence["metrics"]["cv_score"] == 0.82
        assert evidence["strength"] == "strong"

        fetched = store.get_evidence("EE-001")
        assert fetched is not None
        assert fetched["execution_id"] == "E-001"
        assert store.list_evidence()[0]["id"] == "EE-001"
    finally:
        store.close()


def test_belief_update_requires_belief_and_audits(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = ReflectionStore(knowledge, "demo")
    try:
        with pytest.raises(ValueError, match="unknown belief_id"):
            store.append_belief_update(
                belief_id="B-missing",
                prior_confidence=0.4,
                new_confidence=0.55,
            )

        store.ensure_belief("B-001", technique="mixup", confidence=0.4)
        evidence = store.create_evidence(metrics={"cv_score": 0.83})
        update_id = store.append_belief_update(
            belief_id="B-001",
            prior_confidence=0.4,
            new_confidence=0.55,
            prior_status="suggested",
            new_status="validated",
            reason="positive delta vs baseline",
            evidence_id=evidence["id"],
            execution_id="E-001",
        )
        assert update_id >= 1
        updates = store.list_belief_updates("B-001")
        assert len(updates) == 1
        assert updates[0]["new_confidence"] == 0.55
        assert updates[0]["evidence_id"] == "EE-001"
    finally:
        store.close()


def test_claim_and_claim_evidence(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    store = ReflectionStore(knowledge, "demo")
    try:
        evidence = store.create_evidence(strength="strong")
        claim = store.create_claim(
            "Mixup improves CV on this tabular task",
            technique="mixup",
            effect="positive",
            confidence=0.7,
        )
        assert claim["id"] == "C-001"
        assert claim["status"] == "candidate"

        store.link_claim_evidence(claim["id"], evidence["id"], relation="supports")
        edges = store.list_claim_evidence("C-001")
        assert len(edges) == 1
        assert edges[0]["evidence_id"] == evidence["id"]

        lesson = store.create_lesson(
            "Always compare against P-001 baseline before claiming gains",
            category="process",
            confidence=0.6,
        )
        assert lesson["id"] == "L-001"
    finally:
        store.close()
