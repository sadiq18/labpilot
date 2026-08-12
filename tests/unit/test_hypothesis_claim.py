"""`claim_if_proposed` must name exactly one winner (M11 task 7).

K-way fan-out dispatches K branches at once. If two of them are handed the
same hypothesis and both believe they claimed it, the campaign runs the same
experiment twice and — worse — either branch may later `release_claim` a
hypothesis the other is still testing, handing it back to the pool mid-run.
`mark_testing_if_proposed` cannot tell winner from loser; this can.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus


def _store(tmp_path: Path) -> HypothesisStore:
    return HypothesisStore(tmp_path / "knowledge", "titanic")


def _hypothesis(store: HypothesisStore):
    return store.create(observation="a", reason="b", prediction="c", confidence=0.5)


def test_the_first_claim_wins_and_the_second_loses(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)

    first = store.claim_if_proposed(hypothesis.id)
    second = store.claim_if_proposed(hypothesis.id)

    assert first is not None
    assert first.status == HypothesisStatus.TESTING
    assert second is None


def test_exactly_one_of_eight_racing_branches_claims_it(tmp_path: Path) -> None:
    """The K-way case, run for real rather than argued about."""
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)

    results: list[object] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def claim() -> None:
        start.wait()
        outcome = store.claim_if_proposed(hypothesis.id)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [r for r in results if r is not None]
    assert len(winners) == 1
    assert len(results) == 8
    assert store.get(hypothesis.id).status == HypothesisStatus.TESTING


def test_a_non_proposed_hypothesis_cannot_be_claimed(tmp_path: Path) -> None:
    """Confirmed/rejected work is finished — a branch must not re-open it."""
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)
    store.update_status(hypothesis.id, HypothesisStatus.CONFIRMED)

    assert store.claim_if_proposed(hypothesis.id) is None
    assert store.get(hypothesis.id).status == HypothesisStatus.CONFIRMED


def test_a_released_claim_can_be_claimed_again(tmp_path: Path) -> None:
    """The rollback path: setup failed, so the hypothesis returns to the pool."""
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)

    assert store.claim_if_proposed(hypothesis.id) is not None
    store.release_claim(hypothesis.id)

    reclaimed = store.claim_if_proposed(hypothesis.id)
    assert reclaimed is not None
    assert reclaimed.status == HypothesisStatus.TESTING


def test_mark_testing_if_proposed_keeps_its_old_contract(tmp_path: Path) -> None:
    """Its existing caller (reflection/hypotheses/evaluator.py) is unchanged:
    a `Hypothesis` every time, never `None`, whoever got there first.
    """
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)

    first = store.mark_testing_if_proposed(hypothesis.id)
    second = store.mark_testing_if_proposed(hypothesis.id)

    assert first.status == HypothesisStatus.TESTING
    assert second.status == HypothesisStatus.TESTING


def test_a_loser_of_the_race_is_not_told_it_claimed_anything(tmp_path: Path) -> None:
    """The precise failure `mark_testing_if_proposed` has and this does not."""
    store = _store(tmp_path)
    hypothesis = _hypothesis(store)

    store.claim_if_proposed(hypothesis.id)

    # Same call the old method would answer with a TESTING hypothesis.
    assert store.mark_testing_if_proposed(hypothesis.id).status == HypothesisStatus.TESTING
    assert store.claim_if_proposed(hypothesis.id) is None
