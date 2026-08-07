"""Beliefs must heal when the cards behind them are repaired.

Measured on rogii 2026-08-07: `repair_card_directions` re-oriented all 15 cards
and changed **zero** beliefs, because `apply_card_to_beliefs` had already stepped
each one and a step cannot be un-done after the fact. `SWA` — the only technique
that ever improved the metric — stayed recorded as `negative`.
"""

from __future__ import annotations

import json

import pytest

from labpilot.research_engine.evidence.belief_repair import rederive_beliefs_from_cards
from labpilot.research_engine.evidence.models import (
    ClaimEvidenceKind,
    ClaimUpdate,
    EvidenceCard,
    EvidenceDecision,
    ObservedOutcomes,
)
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.paths import ResearchPaths

COMPETITION = "belief-demo"


@pytest.fixture
def knowledge(tmp_path):
    paths = ResearchPaths(tmp_path, COMPETITION).ensure()
    (paths.root / "competition.json").write_text(
        json.dumps({"metric": {"name": "mse", "direction": "minimize"}}), encoding="utf-8"
    )
    return tmp_path


def _card(knowledge_dir, cid_technique, delta, kind, *, decision=EvidenceDecision.ACCEPTED):
    return EvidenceCardStore(knowledge_dir, COMPETITION).save(
        EvidenceCard(
            competition=COMPETITION,
            control_experiment="E-a",
            treatment_experiment="E-b",
            observed=ObservedOutcomes(cv_gain=-1.0, parent_cv=194.8, treatment_cv=193.8),
            technique_attribution={cid_technique: -1.0},
            claim_updates=[
                ClaimUpdate(
                    claim=f"{cid_technique} improves the primary metric",
                    evidence=kind,
                    confidence_delta=delta,
                    technique=cid_technique,
                )
            ],
            decision=decision,
            maximize=False,
        )
    )


def _beliefs(knowledge_dir):
    with KnowledgeStore(knowledge_dir, COMPETITION) as store:
        return {b["technique"]: b for b in store.list_beliefs() if float(b["confidence"] or 0) > 0}


def _seed(knowledge_dir, belief_id, technique, effect, confidence, metadata=None):
    with KnowledgeStore(knowledge_dir, COMPETITION) as store:
        store.upsert_belief(
            belief_id=belief_id,
            technique=technique,
            status="suggested",
            effect=effect,
            confidence=confidence,
            metadata=metadata or {},
        )


# --- rebuilding from cards --------------------------------------------------


def test_a_supporting_card_makes_the_belief_positive(knowledge):
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    assert rederive_beliefs_from_cards(knowledge, COMPETITION)
    swa = _beliefs(knowledge)["SWA"]
    assert swa["effect"] == "positive"
    assert float(swa["confidence"]) == pytest.approx(0.62)


def test_a_contradicting_card_makes_the_belief_negative(knowledge):
    _card(knowledge, "vit", -0.12, ClaimEvidenceKind.CONTRADICT)
    rederive_beliefs_from_cards(knowledge, COMPETITION)
    vit = _beliefs(knowledge)["vit"]
    assert vit["effect"] == "negative"
    assert float(vit["confidence"]) == pytest.approx(0.38)


def test_a_stale_belief_resets_when_its_card_is_retired(knowledge):
    """The vit case: the card behind the belief became `inconclusive`."""
    _seed(
        knowledge,
        f"belief:{COMPETITION}:vit",
        "vit",
        "positive",
        0.62,
        {"last_evidence_card_id": "EV-001"},
    )
    _card(knowledge, "vit", 0.12, ClaimEvidenceKind.SUPPORT, decision=EvidenceDecision.INCONCLUSIVE)
    rederive_beliefs_from_cards(knowledge, COMPETITION)
    vit = _beliefs(knowledge)["vit"]
    assert vit["effect"] == "unknown"
    assert float(vit["confidence"]) == pytest.approx(0.5)


def test_inconclusive_cards_contribute_nothing(knowledge):
    _card(knowledge, "X", 0.12, ClaimEvidenceKind.SUPPORT, decision=EvidenceDecision.INCONCLUSIVE)
    assert rederive_beliefs_from_cards(knowledge, COMPETITION) == []


def test_the_result_depends_only_on_current_cards(knowledge):
    """Recompute, never replay: running twice must not double the confidence."""
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    rederive_beliefs_from_cards(knowledge, COMPETITION)
    first = float(_beliefs(knowledge)["SWA"]["confidence"])
    rederive_beliefs_from_cards(knowledge, COMPETITION)
    assert float(_beliefs(knowledge)["SWA"]["confidence"]) == pytest.approx(first)


def test_repair_is_idempotent(knowledge):
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    _seed(knowledge, "belief_tech_swa", "SWA", "unknown", 0.95)
    assert rederive_beliefs_from_cards(knowledge, COMPETITION)
    assert rederive_beliefs_from_cards(knowledge, COMPETITION) == []


# --- the duplicate identity merge -------------------------------------------


def test_a_literature_belief_stops_outvoting_the_measured_one(knowledge):
    """`belief_tech_swa` at 0.95 vs `belief:comp:swa` at 0.38, same technique.

    Consumers iterate `list_beliefs()` and key by technique, so which row wins
    is an ordering accident. At 0.95 the literature row cleared the promotion
    threshold with `effect: unknown` and was never measured.
    """
    _seed(knowledge, "belief_tech_swa", "SWA", "unknown", 0.95)
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    rederive_beliefs_from_cards(knowledge, COMPETITION)

    with KnowledgeStore(knowledge, COMPETITION) as store:
        rows = {b["id"]: b for b in store.list_beliefs()}
    assert float(rows["belief_tech_swa"]["confidence"]) == 0.0
    assert rows["belief_tech_swa"]["status"] == "superseded"
    assert float(rows[f"belief:{COMPETITION}:swa"]["confidence"]) == pytest.approx(0.62)


def test_the_literature_number_is_kept_not_destroyed(knowledge):
    """Citation count is real information — just not a measurement."""
    _seed(knowledge, "belief_tech_swa", "SWA", "unknown", 0.95)
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    rederive_beliefs_from_cards(knowledge, COMPETITION)

    with KnowledgeStore(knowledge, COMPETITION) as store:
        merged = store.get_belief(f"belief:{COMPETITION}:swa")
        retired = store.get_belief("belief_tech_swa")
    meta = json.loads(merged["metadata"])
    assert meta["literature_confidence"] == 0.95
    assert "belief_tech_swa" in meta["merged_belief_ids"]
    assert json.loads(retired["metadata"])["superseded_by"] == f"belief:{COMPETITION}:swa"


def test_an_unrelated_literature_belief_is_left_alone(knowledge):
    """Only techniques with evidence are touched; the rest of the hub stands."""
    _seed(knowledge, "belief_tech_kriging", "Kriging", "unknown", 0.35)
    _card(knowledge, "SWA", 0.12, ClaimEvidenceKind.SUPPORT)
    rederive_beliefs_from_cards(knowledge, COMPETITION)

    with KnowledgeStore(knowledge, COMPETITION) as store:
        kriging = store.get_belief("belief_tech_kriging")
    assert float(kriging["confidence"]) == pytest.approx(0.35)
    assert kriging["status"] == "suggested"


def test_a_missing_store_does_not_break_a_run(tmp_path):
    assert rederive_beliefs_from_cards(tmp_path / "nope", COMPETITION) == []
