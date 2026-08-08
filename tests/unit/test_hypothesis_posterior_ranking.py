"""Selection must reflect what measurement learned, not what generation guessed.

`_next_hypothesis_id` did sort by `confidence` — that part was never the bug.
The bug is that `confidence` is written once by `HypothesisStore.create` and
updated by **nothing**. It is a prior with no posterior.

Measured on rogii: `hyp:H-010` sat at 0.99 through the runs that disproved it,
and beliefs recorded `SWA | positive | 0.62` beside
`rolling_features | negative | 0.38` while ranking ignored both.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.intelligence.hypothesis.selection import (
    posterior_score,
    rank_hypotheses,
)


class _H:
    def __init__(self, hid, confidence, technique="", stack=(), combos=()):
        self.id = hid
        self.confidence = confidence
        self.technique = technique
        self.technique_stack = list(stack)
        self.combo_techniques = list(combos)


_EFFECTS = {"swa": 0.62, "rollingfeatures": -0.38}


def test_an_untested_hypothesis_keeps_its_prior():
    """Nothing measured means neither promoted nor punished — a fresh pool
    stays ordered by generation's own judgement."""
    assert posterior_score(_H("H-1", 0.7, technique="catboost"), _EFFECTS) == 0.7


def test_a_technique_measured_as_helpful_rises():
    scored = posterior_score(_H("H-1", 0.5, technique="SWA"), _EFFECTS)

    assert scored > 0.5


def test_a_technique_measured_as_harmful_sinks():
    scored = posterior_score(_H("H-1", 0.5, technique="rolling_features"), _EFFECTS)

    assert scored < 0.5


def test_a_confident_but_disproved_hypothesis_ranks_below_an_untried_one():
    """The ordering that matters, and the one rogii got wrong.

    Strong measurement must be able to overturn a confident prior. At the first
    weight I chose this was arithmetically impossible — the swing was bounded
    below the prior gap, so measurement had a ceiling.
    """
    disproved = _H("H-old", 0.9, technique="rolling_features")
    untried = _H("H-new", 0.5, technique="catboost")

    ranked = rank_hypotheses(
        [disproved, untried], "/nowhere", "comp", effects={"rollingfeatures": -0.9}
    )

    assert [h.id for h in ranked] == ["H-new", "H-old"]


def test_weak_evidence_does_not_overturn_a_confident_prior():
    """The other half, and the reason the weight is bounded at all.

    `rolling_features | negative | 0.38` on rogii is a tentative belief. It
    should move the score without erasing everything generation knew — a single
    inconclusive card is not a refutation.
    """
    disproved = _H("H-old", 0.9, technique="rolling_features")
    untried = _H("H-new", 0.5, technique="catboost")

    ranked = rank_hypotheses([disproved, untried], "/nowhere", "comp", effects=_EFFECTS)

    assert [h.id for h in ranked] == ["H-old", "H-new"]


def test_evidence_informs_the_order_without_erasing_the_prior():
    """A single card must not outweigh everything generation knew."""
    strong = _H("H-strong", 0.9, technique="unmeasured")
    weak_but_helpful = _H("H-weak", 0.1, technique="SWA")

    assert posterior_score(strong, _EFFECTS) > posterior_score(weak_but_helpful, _EFFECTS)


def test_a_combo_is_scored_by_mean_not_by_count():
    """Summing would rank a hypothesis naming three techniques above a single,
    on count rather than on evidence."""
    one = _H("H-1", 0.5, technique="SWA")
    three = _H("H-3", 0.5, technique="SWA", stack=["SWA"], combos=["SWA"])

    assert posterior_score(one, _EFFECTS) == pytest.approx(posterior_score(three, _EFFECTS))


def test_a_mixed_claim_nets_out():
    mixed = _H("H-1", 0.5, technique="SWA", stack=["rolling_features"])

    assert posterior_score(mixed, _EFFECTS) == pytest.approx(0.5 + 0.6 * (0.62 - 0.38) / 2)


def test_the_score_stays_within_bounds():
    assert 0.0 <= posterior_score(_H("H-1", 1.0, technique="SWA"), {"swa": 1.0}) <= 1.0
    assert 0.0 <= posterior_score(_H("H-2", 0.0, technique="SWA"), {"swa": -1.0}) <= 1.0


def test_no_beliefs_means_the_prior_order_is_kept():
    """The carve-out must not cost the behaviour it guards."""
    a, b = _H("H-a", 0.4), _H("H-b", 0.8)

    assert [h.id for h in rank_hypotheses([a, b], "/nowhere", "comp")] == ["H-b", "H-a"]


def test_ties_break_deterministically():
    """A selector that varies run to run makes a campaign irreproducible for
    reasons that have nothing to do with research."""
    same = [_H("H-c", 0.5), _H("H-a", 0.5), _H("H-b", 0.5)]

    assert [h.id for h in rank_hypotheses(same, "/nowhere", "comp")] == ["H-a", "H-b", "H-c"]


@pytest.mark.parametrize("bad", [None, "high", object()])
def test_an_unusable_prior_does_not_crash_selection(bad):
    broken = _H("H-1", 0.5)
    broken.confidence = bad

    assert posterior_score(broken, _EFFECTS) == 0.0
