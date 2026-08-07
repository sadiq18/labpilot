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
    """An evidence card from a *genuine* comparison.

    Both `parent_cv` and `treatment_cv` are set, because attribution without
    two real scores is not evidence — see `_card_compared_something_real`. An
    earlier version of this helper omitted them, which made every test here
    exercise a card shape that cannot occur in practice.
    """
    from labpilot.research_engine.evidence.models import EvidenceDecision, ObservedOutcomes

    credit = next(iter(attribution.values()), 0.0)
    parent = 194.80084243002463
    store.save(
        EvidenceCard(
            id=card_id,
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution=attribution,
            # `decision` defaults to `inconclusive`; a card from a real
            # comparison carries a verdict, and the check keys on it.
            decision=EvidenceDecision.ACCEPTED,
            observed=ObservedOutcomes(
                parent_cv=parent, treatment_cv=parent + credit, cv_gain=credit
            ),
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


def test_an_unknown_effect_is_not_a_free_pass(promoter):
    """Rewritten: this test previously asserted the opposite, and was wrong.

    The old reasoning was that "appears to be unknown" asserts nothing about
    impact, so it needs no measurement. But the claim still names a technique
    and still enters research memory, where the Conductor reads it. And the
    beliefs that reach here with `effect="unknown"` are precisely the
    literature-derived ones: `KnowledgeHub._persist_belief` sets confidence from
    citation count (0.95 at five mentions) and never sets an effect. So the
    exemption applied exactly to the beliefs that had never been measured —
    which is how `vit` was promoted on a tabular competition.
    """
    claim = promoter.promote_from_belief(
        {"id": "B-3", "technique": "vit", "effect": "unknown", "confidence": 0.95}
    )
    assert claim is None


def test_a_measured_technique_promotes_regardless_of_effect_wording(promoter, evidence):
    """The guard must gate on measurement, not on the effect string."""
    _card(evidence, "EV-U", {"SWA": -3.83})
    claim = promoter.promote_from_belief(
        {"id": "B-3b", "technique": "SWA", "effect": "unknown", "confidence": 0.95}
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


# --- the assertion is not always in the `effect` column ----------------------


def test_effect_asserted_in_the_statement_is_recognised():
    """The defect that made the first version of this guard useless.

    Two writers disagree about where the assertion lives. Measured 2026-08-07:
    all seven effect-asserting claims on rogii had `effect=''`, so keying on the
    column reached 14 of 417 claims and none of the false ones.
    """
    from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter as CP

    # `_claim_updates_from_attribution` shape — verb in the statement.
    assert CP.asserts_an_effect(
        {"statement": "vit improves the primary metric", "effect": ""}
    )
    assert CP.asserts_an_effect(
        {"statement": "feature_engineering hurts the primary metric", "effect": ""}
    )
    # `promote_from_belief` shape — assertion in the column.
    assert CP.asserts_an_effect(
        {"statement": "SWA appears to be positive on rogii", "effect": "positive"}
    )


def test_a_claim_asserting_nothing_is_left_alone():
    """"appears to be unknown" is 352 of rogii's 417 claims. Contesting those
    would be noise, and would bury the ones that matter."""
    from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter as CP

    assert not CP.asserts_an_effect(
        {"statement": "vit appears to be unknown on rogii", "effect": "unknown"}
    )
    assert not CP.asserts_an_effect({"statement": "", "effect": ""})


def test_the_real_false_claim_is_now_contested(promoter, evidence):
    """rogii's actual row, verbatim: effect empty, assertion in the statement,
    status supported, and both vit runs scored identically to baseline."""
    promoter._reflection.upsert_claim_by_statement(
        statement="vit improves the primary metric",
        technique="vit",
        confidence=0.62,
        status="supported",
        effect="",
    )
    _card(evidence, "EV-20", {"vit": 0.0})
    _card(evidence, "EV-21", {"vit": 0.0})

    contested = promoter.revalidate_claims()

    assert [c["technique"] for c in contested] == ["vit"]
    claims = {c["statement"]: c for c in promoter._reflection.list_claims()}
    assert claims["vit improves the primary metric"]["status"] == "contested"


def test_a_statement_claim_with_real_evidence_survives(promoter, evidence):
    """Control: the broader rule must not sweep up claims that are true."""
    promoter._reflection.upsert_claim_by_statement(
        statement="SWA improves the primary metric",
        technique="SWA",
        confidence=0.8,
        status="supported",
        effect="",
    )
    _card(evidence, "EV-22", {"SWA": -3.826122970779892})
    assert promoter.revalidate_claims() == []


def test_repair_is_reachable_without_a_successful_experiment(tmp_path):
    """The second defect: repair ran only from `record_successful_execution`,
    so a campaign that completed no experiment never repaired itself."""
    from labpilot.research_engine.execution.outcome import revalidate_outcome_claims

    p = ClaimPromoter(tmp_path, COMPETITION)
    p._reflection.upsert_claim_by_statement(
        statement="vit improves the primary metric",
        technique="vit",
        confidence=0.9,
        status="supported",
        effect="",
    )
    p.close()
    EvidenceCardStore(tmp_path, COMPETITION).save(
        EvidenceCard(
            id="EV-23",
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution={"vit": 0.0},
        )
    )

    contested = revalidate_outcome_claims(knowledge_dir=tmp_path, competition=COMPETITION)
    assert [c["technique"] for c in contested] == ["vit"]


def test_repair_never_raises_on_a_broken_store(tmp_path):
    """It runs at campaign start; a failure there must not stop the campaign."""
    from labpilot.research_engine.execution.outcome import revalidate_outcome_claims

    assert revalidate_outcome_claims(knowledge_dir=tmp_path / "nope", competition="x") == []


# --- attribution is only as good as the two scores behind it ----------------


def _card_full(store, card_id, attribution, *, parent, treatment, decision="accepted"):
    from labpilot.research_engine.evidence.models import EvidenceDecision, ObservedOutcomes

    store.save(
        EvidenceCard(
            id=card_id,
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution=attribution,
            decision=EvidenceDecision(decision),
            observed=ObservedOutcomes(
                parent_cv=parent, treatment_cv=treatment,
                cv_gain=(treatment - parent) if None not in (parent, treatment) else None,
            ),
        )
    )


def test_a_control_of_zero_is_a_placeholder_not_a_score(promoter, evidence):
    """rogii's EV-001, verbatim. vit was credited +194.80 — the entire score —
    against `parent_cv=0.0`. No model scores 0.0 on a metric whose baseline is
    ~195; that is a stub run, and it is the sole reason the vit claim read
    `supported`."""
    _card_full(
        evidence, "EV-A", {"vit": 194.80084243002463},
        parent=0.0, treatment=194.80084243002463,
    )

    observations, net = promoter.measured_effect("vit")
    assert (observations, net) == (0, 0.0), "a zero control must not count as evidence"
    assert promoter.effect_is_measured("vit")[0] is False


def test_an_inconclusive_card_does_not_count(promoter, evidence):
    """The evidence builder already labels missing-control comparisons; reuse
    its verdict rather than inventing a second notion of 'real'."""
    _card_full(
        evidence, "EV-B", {"x": 5.0}, parent=194.8, treatment=199.8, decision="inconclusive"
    )
    assert promoter.measured_effect("x") == (0, 0.0)


def test_a_genuine_comparison_still_counts(promoter, evidence):
    """Control: both sides scored, so the gain means something."""
    _card_full(
        evidence, "EV-C", {"SWA": -3.826122970779892},
        parent=194.80084243002463, treatment=190.97471945924474,
    )
    observations, net = promoter.measured_effect("SWA")
    assert observations == 1
    assert net == pytest.approx(-3.826122970779892)
    assert promoter.effect_is_measured("SWA")[0] is True


def test_the_rogii_vit_claim_is_contested_end_to_end(promoter, evidence):
    """Everything together, on the shape the live workspace actually holds."""
    promoter._reflection.upsert_claim_by_statement(
        statement="vit improves the primary metric",
        technique="vit", confidence=0.62, status="supported", effect="",
    )
    promoter._reflection.upsert_claim_by_statement(
        statement="vit appears to be unknown on rogii", technique="vit",
        confidence=0.4, status="candidate", effect="unknown",
    )
    _card_full(evidence, "EV-D", {"vit": 194.8}, parent=0.0, treatment=194.8)

    contested = promoter.revalidate_claims()

    assert len(contested) == 1, "only the claim that asserts an effect"
    by_statement = {c["statement"]: c for c in promoter._reflection.list_claims()}
    assert by_statement["vit improves the primary metric"]["status"] == "contested"
    assert by_statement["vit appears to be unknown on rogii"]["status"] == "candidate"


# --- review findings: PR #95 ------------------------------------------------


def test_claim_phrasing_is_shared_with_the_writer():
    """A wording change in the attribution writer must not silently disable
    revalidation. Importing the constants makes that impossible."""
    from labpilot.research_engine.evidence.builder import CLAIM_HURTS, CLAIM_IMPROVES
    from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter as CP

    assert CP.asserts_an_effect({"statement": f"vit {CLAIM_IMPROVES}", "effect": ""})
    assert CP.asserts_an_effect({"statement": f"vit {CLAIM_HURTS}", "effect": ""})


def test_a_perfect_treatment_score_is_not_discarded(promoter, evidence):
    """`bool(treatment)` would reject a perfect MSE of 0.0 — a real result, not
    a placeholder. Only a zero *control* signals a missing baseline."""
    _card_full(evidence, "EV-E", {"x": -194.8}, parent=194.8, treatment=0.0)
    assert promoter.measured_effect("x")[0] == 1


def test_a_zero_control_is_still_refused(promoter, evidence):
    """The narrow case that matters: gain equals the entire treatment score,
    so there was no baseline to improve on."""
    _card_full(evidence, "EV-F", {"vit": 194.8}, parent=0.0, treatment=194.8)
    assert promoter.measured_effect("vit") == (0, 0.0)


def test_stub_treatment_scores_are_not_caught_here(promoter, evidence):
    """Pins the boundary of this layer, so the docstring cannot overclaim.

    A `treatment_cv=0.5` stub against a 194.8 baseline is indistinguishable
    from a real result *from its scores alone*, so it still counts here. That
    is not the system's only defence: `is_placeholder_metrics` now stops such a
    card being minted, and `repair_card_directions` retires the ones already
    written (see tests/unit/test_placeholder_metrics.py). This test exists so
    that if someone deletes the upstream guard, its absence is visible rather
    than silently covered by a heuristic that was never able to do the job.
    """
    _card_full(evidence, "EV-G", {"stub": -194.3}, parent=194.8, treatment=0.5)
    assert promoter.measured_effect("stub")[0] == 1
