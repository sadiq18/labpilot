"""What the ratchet compares, driven by hand rather than by the corpus.

The integration tests run these functions against the real corpus, which agrees
with its ledger exactly — so both comparisons return empty and a mutation making
them *always* return empty passed the whole file. That is the same trap tier 3
had: the answer you want is the answer that proves nothing.
"""

from __future__ import annotations

from labpilot.accessor.benchmark.ledger import (
    Ledger,
    rates_from,
    regressions,
    unrecorded_gains,
)


def _ledger(**floors: float) -> Ledger:
    return Ledger(corpus_hash="x", floors=floors)


# --- rates ------------------------------------------------------------------


def test_a_rate_counts_only_the_fixtures_that_can_score_it() -> None:
    """`unverifiable` and `not_applicable` are a fixture declining to answer.

    Folding them into the denominator would let a corpus improve its score by
    capturing *less*, which is the one incentive a benchmark must never have.
    """
    rates = rates_from(
        {
            "a": {"target_column": "pass", "metric_name": "unverifiable"},
            "b": {"target_column": "pass", "metric_name": "not_applicable"},
            "c": {"target_column": "fail", "metric_name": "pass"},
        }
    )

    assert rates["target_column"] == 2 / 3
    assert rates["metric_name"] == 1.0, "one fixture could score it, and it passed"


def test_a_known_failure_counts_against_the_rate() -> None:
    """It is a defect the corpus ships red on purpose, not a criterion it cannot
    score — so it belongs in the denominator and not in the numerator."""
    rates = rates_from({"a": {"metric_name": "known_failure"}, "b": {"metric_name": "pass"}})

    assert rates["metric_name"] == 0.5


def test_a_criterion_nothing_can_score_has_no_rate() -> None:
    """Zero out of zero is not zero."""
    assert "metric_name" not in rates_from({"a": {"metric_name": "unverifiable"}})


# --- the two directions -------------------------------------------------------


def test_falling_below_a_floor_is_a_regression() -> None:
    fallen = regressions(_ledger(target_column=1.0), {"target_column": 0.8})

    assert fallen == {"target_column": (1.0, 0.8)}


def test_meeting_the_floor_exactly_is_not_a_regression() -> None:
    assert regressions(_ledger(target_column=1.0), {"target_column": 1.0}) == {}


def test_rising_above_a_floor_is_an_unrecorded_gain() -> None:
    """Silently absorbing it is how a ratchet rots: the floor stays where it was
    and the next regression falls into the slack without tripping anything."""
    gained = unrecorded_gains(_ledger(metric_name=0.75), {"metric_name": 1.0})

    assert gained == {"metric_name": (0.75, 1.0)}


def test_a_criterion_with_no_floor_is_neither() -> None:
    """A newly added criterion has nowhere to fall from, and treating its absence
    as zero would make every addition look like a win."""
    ledger = _ledger(target_column=1.0)

    assert regressions(ledger, {"target_column": 1.0, "brand_new": 0.0}) == {}
    assert unrecorded_gains(ledger, {"target_column": 1.0, "brand_new": 1.0}) == {}


def test_a_floor_for_a_criterion_the_run_did_not_produce_is_not_a_regression() -> None:
    """A criterion that vanished from the run is a different problem — a missing
    scorer, not a worse one — and reporting it as a fall would send the reader to
    the wrong place."""
    assert regressions(_ledger(gone=1.0), {"target_column": 1.0}) == {}
