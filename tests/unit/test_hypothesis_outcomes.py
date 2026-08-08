"""A failed attempt must say whether trying again could ever help.

Two failures found on rogii 2026-08-09, and they are the same missing idea:

* a hypothesis leaves `testing` **only** when an evidence card is written, and a
  failed execution writes none — so a failed hypothesis stayed `testing` forever,
  out of the pool and never retired. One stuck, three historically.
* a hypothesis whose change was already implemented stayed `proposed` and was
  re-selected on every step of four consecutive campaigns.

The rule is the one `BaseMicroAgent.run` already follows for LLM calls, one
layer up: retry transient failures, and record a failure only when attempts are
exhausted.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.reflection.hypotheses.outcomes import (
    DEFAULT_MAX_ATTEMPTS,
    HypothesisOutcome,
    classify_hypothesis_failure,
)


def test_a_redundant_hypothesis_is_a_dead_end_immediately():
    """The rogii case. A change already present in the parent will be present on
    every future attempt, so retrying is guaranteed waste."""
    outcome, why = classify_hypothesis_failure(
        redundant=True, failure_reason="already implemented: the parent calls 'lgb'"
    )

    assert outcome is HypothesisOutcome.DEAD_END
    assert "lgb" in why


def test_redundancy_outranks_a_healthy_attempt_count():
    """One attempt is enough to settle it — waiting for three would spend two
    campaign steps to reach a conclusion already proven."""
    outcome, _ = classify_hypothesis_failure(redundant=True, attempts=1)

    assert outcome is HypothesisOutcome.DEAD_END


@pytest.mark.parametrize("kind", ["rate_limit", "unavailable", "timeout", "no_client"])
def test_infrastructure_failures_are_retryable(kind):
    """None of these is evidence about the idea being tested."""
    outcome, why = classify_hypothesis_failure(failure_kind=kind, attempts=1)

    assert outcome is HypothesisOutcome.RETRYABLE
    assert kind in why


def test_an_unknown_failure_is_retryable_not_fatal():
    """An unrecognised error is not evidence against the idea. Retiring on one
    would discard hypotheses for defects in the harness — which this system has
    done before, recording `SWA` as harmful when the fault was metric direction.
    """
    outcome, _ = classify_hypothesis_failure(failure_kind="something_new", attempts=1)

    assert outcome is HypothesisOutcome.RETRYABLE


def test_exhaustion_retires_even_a_transient_failure():
    """A rate limit that has blocked three attempts is still a campaign making
    no progress. Treating "transient" as "retry forever" is how a loop with a
    plausible excuse runs to its step budget."""
    outcome, why = classify_hypothesis_failure(
        failure_kind="rate_limit", attempts=DEFAULT_MAX_ATTEMPTS
    )

    assert outcome is HypothesisOutcome.DEAD_END
    assert "3 time(s)" in why


def test_the_last_attempt_before_the_budget_is_still_retryable():
    outcome, _ = classify_hypothesis_failure(
        failure_kind="rate_limit", attempts=DEFAULT_MAX_ATTEMPTS - 1
    )

    assert outcome is HypothesisOutcome.RETRYABLE


def test_a_retirement_always_carries_its_reason():
    """A retired hypothesis whose retirement is unexplained is indistinguishable
    from one that was never good — the first is a finding, the second noise."""
    outcome, why = classify_hypothesis_failure(
        failure_reason="smoke gate failed", failure_kind="other", attempts=9
    )

    assert outcome is HypothesisOutcome.DEAD_END
    assert "smoke gate failed" in why


def test_a_retirement_with_no_recorded_reason_still_says_something():
    _, why = classify_hypothesis_failure(attempts=9)

    assert why.strip()
