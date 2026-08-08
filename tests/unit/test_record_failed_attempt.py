"""The transition that unsticks a hypothesis after its experiment fails.

Against a real `HypothesisStore`, because the bug being fixed is a *state
machine* gap: a hypothesis leaves `testing` only when an evidence card is
written, and a failed execution writes none.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.reflection.hypotheses.evaluator import HypothesisEvaluator
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

_COMP = "rogii-wellbore-geology-prediction"


def _evaluator(tmp_path) -> HypothesisEvaluator:
    return HypothesisEvaluator(tmp_path / "knowledge", _COMP)


def _seed(evaluator, status: HypothesisStatus) -> str:
    """Create through the store's own API, then move it to `status`.

    `create` allocates the id and starts at `proposed`, so the status is reached
    by the same transition production uses rather than by writing a file behind
    the store's back — a fixture that bypasses the store would not prove the
    store agrees.
    """
    hypothesis = evaluator.store.create(observation="o", reason="r", prediction="p", confidence=0.5)
    if status is not HypothesisStatus.PROPOSED:
        evaluator.store.update_status(hypothesis.id, status)
    return hypothesis.id


def test_a_transient_failure_returns_it_to_the_pool(tmp_path):
    """Otherwise it sits in `testing` forever: out of the pool, never retried,
    never retired. Measured on rogii — one stuck, three historically."""
    ev = _evaluator(tmp_path)
    hid = _seed(ev, HypothesisStatus.TESTING)

    result = ev.record_failed_attempt(hid, failure_kind="rate_limit", attempts=1)

    assert result.status is HypothesisStatus.PROPOSED
    assert "rate_limit" in (result.actual_outcome or "")


def test_a_redundant_hypothesis_is_retired_with_its_reason(tmp_path):
    """The rogii loop: re-selected on every step of four campaigns."""
    ev = _evaluator(tmp_path)
    hid = _seed(ev, HypothesisStatus.TESTING)

    result = ev.record_failed_attempt(
        hid, redundant=True, failure_reason="the parent already calls 'lgb'"
    )

    assert result.status is HypothesisStatus.REJECTED
    assert "lgb" in (result.actual_outcome or "")


def test_a_retired_hypothesis_is_not_selected_again(tmp_path):
    """`_next_hypothesis_id` lists only `proposed`, so rejection is what removes
    it from selection — the property the whole fix exists for."""
    ev = _evaluator(tmp_path)
    hid = _seed(ev, HypothesisStatus.TESTING)
    ev.record_failed_attempt(hid, redundant=True)

    still_open = ev.store.list(status=HypothesisStatus.PROPOSED)
    assert [h.id for h in still_open] == []


def test_exhaustion_retires_it(tmp_path):
    ev = _evaluator(tmp_path)
    hid = _seed(ev, HypothesisStatus.TESTING)

    result = ev.record_failed_attempt(hid, failure_kind="rate_limit", attempts=3)

    assert result.status is HypothesisStatus.REJECTED


@pytest.mark.parametrize("settled", [HypothesisStatus.CONFIRMED, HypothesisStatus.REJECTED])
def test_a_settled_verdict_is_never_demoted(settled, tmp_path):
    """A verdict reached by *measurement* outranks a failed attempt. Returning a
    confirmed hypothesis to `proposed` would re-queue work already answered."""
    ev = _evaluator(tmp_path)
    hid = _seed(ev, settled)

    result = ev.record_failed_attempt(hid, failure_kind="timeout", attempts=1)

    assert result.status is settled


def test_an_unknown_hypothesis_is_not_an_error(tmp_path):
    """Bookkeeping must never kill a run."""
    assert _evaluator(tmp_path).record_failed_attempt("H-nope") is None


def test_an_empty_id_is_ignored(tmp_path):
    assert _evaluator(tmp_path).record_failed_attempt("") is None
