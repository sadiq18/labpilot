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

COMPETITION = "vocab-demo"


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
        history_cols = {
            row[1]
            for row in client.conn.execute(
                "PRAGMA table_info(technique_status_history)"
            ).fetchall()
        }
        assert "signed_net" in history_cols
        assert "net_effect" not in history_cols
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
            status, reason, obs, *_ = derive_technique_status(name, promoter)
            assert status == "candidate"
            assert obs == 0
            assert "no conclusive evidence" in reason
    finally:
        promoter.close()


def test_swa_is_confirmed_on_mse_improvement(tmp_path) -> None:
    """Negative raw credit on MSE is an improvement once ``maximize=False``."""
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["SWA"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-swa", {"SWA": -3.83}, maximize=False)

    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, _, obs, raw_net, signed_net, _ = derive_technique_status("SWA", promoter)
        assert status == "confirmed"
        assert obs == 1
        assert raw_net == pytest.approx(-3.83)
        assert signed_net == pytest.approx(3.83)
    finally:
        promoter.close()


def test_zero_observation_is_never_confirmed_despite_high_confidence(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        store.merge_technique("vit", confidence=0.95)
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, _, obs, *_ = derive_technique_status("vit", promoter)
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
        status, reason, obs, *_ = derive_technique_status("vit", promoter)
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
    _card(cards, "EV-swa", {"SWA": -3.83}, maximize=False)
    recompute_technique_status(tmp_path, COMPETITION)
    (cards.dir / "EV-swa.json").unlink()
    recompute_technique_status(tmp_path, COMPETITION)

    db_path = ResearchPaths(tmp_path, COMPETITION).db_path
    with SqliteClient(db_path) as client:
        rows = client.conn.execute(
            "SELECT from_status, to_status, signed_net "
            "FROM technique_status_history WHERE technique_id = ?",
            (tid,),
        ).fetchall()
    assert len(rows) >= 2
    assert any(r["to_status"] == "confirmed" for r in rows)
    assert rows[-1]["to_status"] == "candidate"
    confirmed = next(r for r in rows if r["to_status"] == "confirmed")
    assert confirmed["signed_net"] == pytest.approx(3.83)


def test_invalid_status_is_rejected(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        tid = store.merge_technique("vit")
        with pytest.raises(ValueError, match="invalid technique status"):
            store.set_technique_status(tid, "bogus", competition=COMPETITION, reason="test")


def test_report_lists_would_change_without_apply(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["SWA", "the"])
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-swa", {"SWA": -3.83}, maximize=False)

    report = technique_status_report(tmp_path, COMPETITION)
    assert report["total"] == 2
    assert report["counts"]["confirmed"] == 1
    assert report["counts"]["candidate"] == 1
    assert report["counts"]["dormant"] == 0
    assert len(report["would_change"]) == 1
    assert report["would_change"][0]["name"] == "SWA"
    assert report["would_change"][0]["signed_net"] == pytest.approx(3.83)


def test_unselected_stays_candidate_until_aged(tmp_path) -> None:
    """Aging is what makes dormant safe — first recompute must not close the vocab."""
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["gradient_boosting_dart"])
        created = store.list_techniques()[0]["created_at"]
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, reason, *_ = derive_technique_status(
            "gradient_boosting_dart",
            promoter,
            selected=set(),
            created_at=created,
            session_times=[],
        )
        assert status == "candidate"
        assert "never selected" not in reason
    finally:
        promoter.close()


def test_unselected_becomes_dormant_after_n_campaigns(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["3D garment modeling"])
        created = store.list_techniques()[0]["created_at"]
    later = [
        "2099-01-01T00:00:00+00:00",
        "2099-01-02T00:00:00+00:00",
    ]
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, reason, obs, *_ = derive_technique_status(
            "3D garment modeling",
            promoter,
            selected=set(),
            created_at=created,
            session_times=later,
            dormant_after=2,
        )
        assert status == "dormant"
        assert obs == 0
        assert "2 campaign" in reason
    finally:
        promoter.close()


def test_fresh_technique_after_campaigns_stays_candidate(tmp_path) -> None:
    """A technique proposed *after* existing campaigns must remain visible."""
    sessions = ["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"]
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["brand_new_trick"])
        created = store.list_techniques()[0]["created_at"]
    assert created > sessions[-1]
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, *_ = derive_technique_status(
            "brand_new_trick",
            promoter,
            selected=set(),
            created_at=created,
            session_times=sessions,
            dormant_after=2,
        )
        assert status == "candidate"
    finally:
        promoter.close()


def test_selected_unmeasured_never_goes_dormant(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        _seed_techniques(store, ["waiting"])
        created = store.list_techniques()[0]["created_at"]
    later = ["2099-01-01T00:00:00+00:00", "2099-01-02T00:00:00+00:00"]
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, *_ = derive_technique_status(
            "waiting",
            promoter,
            selected={"waiting"},
            created_at=created,
            session_times=later,
            dormant_after=2,
        )
        assert status == "candidate"
    finally:
        promoter.close()


def test_campaigns_since_parses_mixed_iso_forms() -> None:
    """Z vs +00:00 must not invert ordering via lexicographic string compare."""
    from labpilot.research_engine.execution.technique.vocabulary import campaigns_since

    assert (
        campaigns_since(
            "2026-08-07T15:38:16.500000+00:00",
            ["2026-08-07T15:38:16Z", "2026-08-07T15:38:17Z"],
        )
        == 1
    )
    assert (
        campaigns_since(
            "2026-08-07T15:38:16Z",
            ["2026-08-07T15:38:16.999999+00:00"],
        )
        == 1
    )

def test_selection_matches_across_spacing_and_underscores(tmp_path) -> None:
    """Hypothesis 'Grad Boost' must keep vocabulary 'grad_boost' from dormant."""
    from labpilot.research_engine.execution.technique.vocabulary import _vocab_key

    with KnowledgeStore(tmp_path, COMPETITION) as store:
        store.merge_technique("grad_boost")
        created = store.list_techniques()[0]["created_at"]
    later = ["2099-01-01T00:00:00+00:00", "2099-01-02T00:00:00+00:00"]
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        status, *_ = derive_technique_status(
            "grad_boost",
            promoter,
            selected={_vocab_key("Grad Boost")},
            created_at=created,
            session_times=later,
            dormant_after=2,
        )
        assert status == "candidate"
    finally:
        promoter.close()
