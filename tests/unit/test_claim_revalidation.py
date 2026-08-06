"""A claim must be backed by a measurement, not by confidence alone.

Measured on rogii 2026-08-07. Both `vit` experiments scored
194.80084243002463 — byte-identical to the untouched baseline — and the system
recorded:

    "vit improves the primary metric"   status=supported   confidence=0.62

`evidence/builder.py` already refuses to mint a claim for a zero-credit
technique (`if abs(credit) < 1e-9: continue`). `ClaimPromoter` promoted on
confidence alone, so the rule existed in one place and was missing in the
other — the same shape as every other defect found that day.

Confidence is the wrong gate on its own because it is produced by the loop that
consumes the claim: a belief raises its own confidence, crosses the threshold,
and becomes a "finding" nothing measured ever supported. 45 such claims kept the
Conductor proposing vit for a tabular regression.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.evidence.models import EvidenceCard
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

COMPETITION = "revalidation-demo"


def _card(store: EvidenceCardStore, card_id: str, attribution: dict[str, float]) -> None:
    store.save(
        EvidenceCard(
            id=card_id,
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution=attribution,
        )
    )


@pytest.fixture
def promoter(tmp_path):
    p = ClaimPromoter(tmp_path, COMPETITION)
    yield p
    p.close()


@pytest.fixture
def evidence(tmp_path):
    store = EvidenceCardStore(tmp_path, COMPETITION)
    yield store


# --- the measurement itself -------------------------------------------------


def test_no_evidence_means_no_measured_effect(promoter):
    ok, why = promoter.effect_is_measured("vit")
    assert ok is False
    assert "no evidence card" in why


def test_zero_credit_is_not_an_effect(promoter, evidence):
    """rogii's exact case: the technique ran and changed nothing."""
    _card(evidence, "EV-1", {"vit": 0.0})
    _card(evidence, "EV-2", {"vit": 0.0})

    ok, why = promoter.effect_is_measured("vit")
    assert ok is False
    assert "~0" in why, "with several runs the reason should say the net is zero"
    assert "2 run(s)" in why, "the reason should say how much evidence was weighed"


def test_a_real_change_counts_whichever_direction_it_points(promoter, evidence):
    """`SWA` scored **-3.83** for a genuine improvement on MSE, so a check that
    inferred 'positive' from the sign would be wrong half the time. Only
    'no effect at all' is direction-agnostic."""
    _card(evidence, "EV-3", {"SWA": -3.826122970779892})
    ok, _ = promoter.effect_is_measured("SWA")
    assert ok is True

    _card(evidence, "EV-4", {"target_encoding": 0.004})
    assert promoter.effect_is_measured("target_encoding")[0] is True


def test_offsetting_runs_net_to_no_effect(promoter, evidence):
    """Two runs that cancel out have measured nothing about the technique.

    The reason must not say "changed nothing" — these runs each measured a real
    change, and they disagree. Overstating that would be its own false finding.
    """
    _card(evidence, "EV-5", {"x": 1.5})
    _card(evidence, "EV-6", {"x": -1.5})
    ok, why = promoter.effect_is_measured("x")
    assert ok is False
    assert "~0" in why and "2 run(s)" in why
    assert "no run measured any change" not in why


# --- the gate ---------------------------------------------------------------


def test_a_high_confidence_belief_is_not_promoted_without_measurement(promoter, evidence):
    """The bug. Confidence well above threshold, effect asserted, nothing measured."""
    _card(evidence, "EV-7", {"vit": 0.0})
    claim = promoter.promote_from_belief(
        {"id": "B-1", "technique": "vit", "effect": "positive", "confidence": 0.95}
    )
    assert claim is None


def test_the_same_belief_promotes_once_a_measurement_exists(promoter, evidence):
    """Control: the gate must not block everything, or it is indistinguishable
    from switching promotion off."""
    _card(evidence, "EV-8", {"SWA": -3.83})
    claim = promoter.promote_from_belief(
        {"id": "B-2", "technique": "SWA", "effect": "positive", "confidence": 0.95}
    )
    assert claim is not None
    assert claim["technique"] == "SWA"


def test_an_unknown_effect_is_still_promotable(promoter):
    """"appears to be unknown" asserts nothing about impact, so it needs no
    measurement to back it."""
    claim = promoter.promote_from_belief(
        {"id": "B-3", "technique": "vit", "effect": "unknown", "confidence": 0.95}
    )
    assert claim is not None


# --- self-healing -----------------------------------------------------------


def test_existing_unsupported_claims_are_contested_not_deleted(promoter, evidence):
    """The system repairs its own memory rather than needing a hand-run script.

    Contested, never deleted: what the system once believed — and why it
    stopped — is itself research evidence.
    """
    _card(evidence, "EV-9", {"SWA": -3.83})
    supported = promoter.promote_from_belief(
        {"id": "B-4", "technique": "SWA", "effect": "positive", "confidence": 0.95}
    )
    assert supported is not None

    # A claim that no measurement backs, written before the rule existed.
    promoter._reflection.upsert_claim_by_statement(
        statement="vit improves the primary metric",
        technique="vit",
        confidence=0.62,
        status="supported",
        effect="positive",
    )
    _card(evidence, "EV-10", {"vit": 0.0})

    contested = promoter.revalidate_claims()

    assert [c["technique"] for c in contested] == ["vit"]
    claims = {c["statement"]: c for c in promoter._reflection.list_claims()}
    assert claims["vit improves the primary metric"]["status"] == "contested"
    assert len(claims) == 2, "nothing was deleted"


def test_revalidation_leaves_supported_claims_alone(promoter, evidence):
    _card(evidence, "EV-11", {"SWA": -3.83})
    promoter.promote_from_belief(
        {"id": "B-5", "technique": "SWA", "effect": "positive", "confidence": 0.95}
    )
    assert promoter.revalidate_claims() == []


def test_promotion_repairs_before_it_adds(promoter, evidence):
    """`promote_eligible` revalidates first, so a campaign self-heals on its
    next cycle without anyone running a migration."""
    promoter._reflection.upsert_claim_by_statement(
        statement="vit improves the primary metric",
        technique="vit",
        confidence=0.9,
        status="supported",
        effect="positive",
    )
    _card(evidence, "EV-12", {"vit": 0.0})

    promoter.promote_eligible()

    claims = {c["statement"]: c for c in promoter._reflection.list_claims()}
    assert claims["vit improves the primary metric"]["status"] == "contested"
