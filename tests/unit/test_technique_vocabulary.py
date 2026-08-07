"""Technique vocabulary status — derived from evidence cards (M-25 step 1)."""

from __future__ import annotations

import pytest

from labpilot.accessor.sqlite import SCHEMA_VERSION, SqliteClient
from labpilot.research_engine.evidence.models import EvidenceCard, EvidenceDecision, ObservedOutcomes
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.execution.technique.vocabulary import (
    derive_technique_status,
    recompute_technique_status,
    technique_status_report,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

COMPETITION = "vocab-demo"


def _select_technique(knowledge_dir, technique: str) -> None:
    HypothesisStore(knowledge_dir, COMPETITION).create(
        observation=f"Try {technique}",
        reason="unit test selection",
        prediction="measurable",
        confidence=0.5,
        technique=technique,
    )


def _seed_techniques(store: KnowledgeStore, names: list[str]) -> None:
    for name in names:
        store.merge_technique(name)


def _card(
    store: EvidenceCardStore,
    card_id: str,
    attribution: dict[str, float],
    *,
    maximize: bool = False,
) -> None:
    credit = next(iter(attribution.values()), 0.0)
    parent = 194.8
    store.save(
        EvidenceCard(
            id=card_id,
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution=attribution,
            decision=EvidenceDecision.ACCEPTED,
            maximize=maximize,
            observed=ObservedOutcomes(
                parent_cv=parent,
                treatment_cv=parent + credit,
                cv_gain=credit,
            ),
        )
    )


def test_migration_adds_technique_status_column(tmp_path) -> None:
    db = tmp_path / "knowledge.db"
    client = SqliteClient(db)
    try:
        assert client.schema_version() == SCHEMA_VERSION
        cols = {
            row[1]
            for row in client.conn.execute("PRAGMA table_info(techniques)").fetchall()
        }
        assert "status" in cols
        tables = {
            row["name"]
            for row in client.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "technique_status_history" in tables
    finally:
        client.close()


def test_migration_upgrades_legacy_techniques_table(tmp_path) -> None:
    """Pre-v11 DBs lack ``techniques.status`` — index creation must not run first."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE techniques (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL DEFAULT '',
            domain TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            known_issues TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.close()

    client = SqliteClient(db)
    try:
        cols = {
            row[1]
            for row in client.conn.execute("PRAGMA table_info(techniques)").fetchall()
        }
        assert "status" in cols
        assert client.schema_version() == SCHEMA_VERSION
    finally:
        client.close()


def test_junk_stays_candidate_without_evidence(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["the", "Breath Focus practice", "3D garment modeling"])
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        for name in ("the", "Breath Focus practice", "3D garment modeling"):
            status, reason, obs, _, _ = derive_technique_status(name, promoter)
            assert status == "candidate"
            assert obs == 0
            assert "no conclusive evidence" in reason
    finally:
        promoter.close()


def test_unselected_unmeasured_becomes_dormant(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["3D garment modeling"])
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, reason, obs, _, _ = derive_technique_status(
            "3D garment modeling", promoter, selected=set()
        )
        assert status == "dormant"
        assert obs == 0
        assert "never selected" in reason
    finally:
        promoter.close()


def test_swa_is_confirmed_on_mse_improvement(tmp_path) -> None:
    """Negative raw credit on MSE is an improvement once ``maximize=False``."""
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["SWA"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-swa", {"SWA": -3.83}, maximize=False)

    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, _, obs, net, _ = derive_technique_status("SWA", promoter)
        assert status == "confirmed"
        assert obs == 1
        assert net == pytest.approx(-3.83)
    finally:
        promoter.close()


def test_zero_observation_is_never_confirmed_despite_high_confidence(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        store.merge_technique("vit", confidence=0.95)
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, _, obs, _, _ = derive_technique_status("vit", promoter)
        assert status == "candidate"
        assert obs == 0
    finally:
        promoter.close()


def test_harmful_technique_is_rejected(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["vit"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-vit", {"vit": 5.0}, maximize=False)

    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, reason, obs, _, _ = derive_technique_status("vit", promoter)
        assert status == "rejected"
        assert obs == 1
        assert "adverse" in reason
    finally:
        promoter.close()


def test_recompute_is_idempotent(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["SWA", "the"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-swa", {"SWA": -3.83}, maximize=False)

    first = recompute_technique_status(tmp_path, COMPETITION)
    second = recompute_technique_status(tmp_path, COMPETITION)
    assert first
    assert second == []


def test_retiring_evidence_demotes_to_candidate(tmp_path) -> None:
    cards = EvidenceCardStore(tmp_path, COMPETITION)
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        tid = store.merge_technique("vit")
    _select_technique(tmp_path, "vit")
    _card(cards, "EV-vit", {"vit": -2.0}, maximize=False)
    recompute_technique_status(tmp_path, COMPETITION)
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        assert store.get_technique(tid)["status"] == "confirmed"

    (cards.dir / "EV-vit.json").unlink()
    recompute_technique_status(tmp_path, COMPETITION)
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        assert store.get_technique(tid)["status"] == "candidate"


def test_history_survives_demotion(tmp_path) -> None:
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    cards = EvidenceCardStore(tmp_path, COMPETITION)
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        tid = store.merge_technique("SWA")
    _select_technique(tmp_path, "SWA")
    _card(cards, "EV-swa", {"SWA": -3.83}, maximize=False)
    recompute_technique_status(tmp_path, COMPETITION)
    (cards.dir / "EV-swa.json").unlink()
    recompute_technique_status(tmp_path, COMPETITION)

    db_path = ResearchPaths(tmp_path, COMPETITION).db_path
    with SqliteClient(db_path) as client:
        rows = client.conn.execute(
            "SELECT from_status, to_status FROM technique_status_history WHERE technique_id = ?",
            (tid,),
        ).fetchall()
    assert len(rows) >= 2
    assert any(r["to_status"] == "confirmed" for r in rows)
    assert rows[-1]["to_status"] == "candidate"


def test_claim_promotion_blocks_rejected_and_needs_measurement(tmp_path) -> None:
    """Rejected never promotes; confirmed status alone is not enough without measurement."""
    cards = EvidenceCardStore(tmp_path, COMPETITION)
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        tid = store.merge_technique("vit", confidence=0.95)
        store.set_technique_status(
            tid,
            "rejected",
            competition=COMPETITION,
            reason="test",
        )
        store.upsert_belief(
            belief_id="belief:vit",
            technique="vit",
            status="validated",
            effect="positive",
            confidence=0.95,
        )
    _card(cards, "EV-vit-bad", {"vit": 5.0}, maximize=False)
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        belief = promoter._knowledge.list_beliefs()[0]
        assert promoter.promote_from_belief(belief) is None
        with KnowledgeStore(tmp_path, COMPETITION) as store:
            store.set_technique_status(
                tid,
                "confirmed",
                competition=COMPETITION,
                from_status="rejected",
                reason="test",
            )
        # Confirmed + measurement would promote; strip the card first.
        (cards.dir / "EV-vit-bad.json").unlink()
        assert promoter.promote_from_belief(belief) is None
    finally:
        promoter.close()


def test_report_lists_would_change_without_apply(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["SWA", "the"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-swa", {"SWA": -3.83}, maximize=False)

    report = technique_status_report(tmp_path, COMPETITION)
    assert report["total"] == 2
    assert report["counts"]["confirmed"] == 1
    assert report["counts"]["dormant"] == 1
    assert len(report["would_change"]) == 2
    names = {row["name"] for row in report["would_change"]}
    assert names == {"SWA", "the"}
