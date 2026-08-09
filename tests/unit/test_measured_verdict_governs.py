"""A hypothesis cannot be confirmed by a regression.

Measured on rogii 2026-08-09. Execution E-234 raised `cv_rmse` from 194 to
1382 — seven times worse — `comparison.json` recorded `decision: "rejected"`,
and `H-096` was written **confirmed**.

Two defects compounded. `_map_outcomes` read `comparison["verdict"]`, a key
nothing writes (the comparator writes `decision` and `outcome`), so the one
verdict derived from measurement never arrived. It then fell through to a
heuristic reading the sign of `cv_delta` as though larger were always better,
and `+1188` on an error metric was taken for an improvement.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.reflection.critic.critic import ExperimentCritic


def _assess(comparison: dict, **evidence):
    return ExperimentCritic().assess({"comparison": comparison, **evidence})


def test_the_measured_rejection_is_not_overturned():
    """The exact E-234 shape."""
    result = _assess(
        {"decision": "rejected", "outcome": "rejected", "delta": 1188.07},
    )

    assert result.hypothesis_outcome == "rejected"
    assert result.belief_effect == "contradicts"


@pytest.mark.parametrize("key", ["decision", "outcome", "verdict"])
def test_every_key_the_comparator_might_write_is_read(key):
    """`verdict` was the only one read and is written by nothing."""
    assert _assess({key: "rejected"}).hypothesis_outcome == "rejected"


def test_a_measured_acceptance_still_confirms():
    """The carve-out must not cost the behaviour it guards."""
    assert _assess({"decision": "worth_keeping"}).hypothesis_outcome == "confirmed"


def test_a_positive_delta_alone_confirms_nothing():
    """Nothing at this layer knows the metric's direction, so it no longer
    guesses. `+1188` was read as an improvement because larger was assumed
    better; for RMSE it is the opposite."""
    result = _assess({"delta": 1188.07})

    assert result.hypothesis_outcome == "inconclusive"


def test_no_comparison_withholds_judgement():
    """A missing comparison is a reason to withhold a verdict, not to invent
    one. `inconclusive` costs a re-test; `confirmed` on a regression poisons
    every ranking that reads it afterwards."""
    assert _assess({}).hypothesis_outcome == "inconclusive"
