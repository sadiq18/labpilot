"""Counting work worth doing, not rows in a table.

`untested_hypothesis_count` counted every `proposed` row, and that number was
the only thing standing between a campaign and fresh evidence. Measured on rogii
2026-08-09: **46 proposed**, 43 generated on 2026-08-07 and never selected,
including `3D garment modeling` and `Breath Focus practice` for a wellbore
regression. Each held the fetch gate shut as firmly as a good idea would.

Stale here means *never chosen*, not merely old — a hypothesis the selector has
passed over for two campaigns has been rejected in practice, once per step.
Reuses M18's `campaigns_since` rather than a second staleness clock.
"""

from __future__ import annotations

from labpilot.research_engine.intelligence.hypothesis.viability import (
    STALE_AFTER_SELECTIONS,
    viable_hypothesis_count,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

_COMP = "rogii-wellbore-geology-prediction"

#: A selection well after anything these tests create, so aging is decided by
#: *which* selections count, never by clock skew between them.
_STAMP = "2099-01-01T00:00:00+00:00"


def _select(tmp_path, *hypothesis_ids: str) -> None:
    """Mint a plan against each hypothesis — one selection apiece."""
    from datetime import UTC, datetime

    from labpilot.research_engine.planner.schemas.models import ResearchPlan
    from labpilot.research_engine.planner.schemas.task_types import PlanStatus
    from labpilot.research_engine.planner.store import PlanStore

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        for index, hypothesis_id in enumerate(hypothesis_ids, start=1):
            now = datetime.now(UTC)
            store.upsert_plan(
                ResearchPlan(
                    id=f"P-{index:03d}",
                    competition=_COMP,
                    hypothesis_id=hypothesis_id,
                    goal="g",
                    status=PlanStatus.READY,
                    created_at=now,
                    updated_at=now,
                )
            )
    finally:
        store.close()


def _count(tmp_path) -> int:
    return viable_hypothesis_count(tmp_path / "knowledge", _COMP)


def _store(tmp_path) -> HypothesisStore:
    return HypothesisStore(tmp_path / "knowledge", _COMP)


def _propose(store, n: int = 1) -> list[str]:
    return [
        store.create(observation="o", reason="r", prediction="p", confidence=0.5).id
        for _ in range(n)
    ]


def test_fresh_proposals_all_count(tmp_path):
    _propose(_store(tmp_path), 4)
    assert _count(tmp_path) == 4


def test_an_empty_store_counts_zero(tmp_path):
    assert _count(tmp_path) == 0


def test_a_missing_store_counts_zero_and_opens_the_gate(tmp_path):
    """Zero means "gather", which is the safe direction: the failure this
    module exists to prevent is a gate stuck shut."""
    assert viable_hypothesis_count(tmp_path / "nowhere", _COMP) == 0


def test_tested_hypotheses_do_not_count(tmp_path):
    """They have left `proposed` entirely — the count is of *pending* work."""
    store = _store(tmp_path)
    ids = _propose(store, 3)
    store.update_status(ids[0], HypothesisStatus.TESTING)
    store.update_status(ids[1], HypothesisStatus.REJECTED)

    assert _count(tmp_path) == 1


def test_a_retired_hypothesis_stops_holding_the_gate_shut(tmp_path):
    """The point of step 2: retiring one has to move this number."""
    store = _store(tmp_path)
    ids = _propose(store, 6)
    before = _count(tmp_path)
    store.update_status(ids[0], HypothesisStatus.REJECTED)

    assert _count(tmp_path) == before - 1


def test_nothing_selected_means_nothing_is_stale(tmp_path):
    """A workspace where the selector has never chosen has declined nothing.

    Campaign count was the first measure and was too generous: a campaign that
    crashed at step three never chose anything, so counting it as a rejection
    punished a hypothesis for an infrastructure failure.
    """
    _propose(_store(tmp_path), 3)

    assert _count(tmp_path) == 3


def test_the_threshold_matches_the_technique_clock():
    """Two independent staleness rules would drift and disagree about the same
    workspace."""
    from labpilot.research_engine.execution.technique.status_constants import (
        DORMANT_AFTER_CAMPAIGNS,
    )

    assert STALE_AFTER_SELECTIONS == DORMANT_AFTER_CAMPAIGNS


def test_counting_never_changes_a_status(tmp_path):
    """Stale rows are not retired — this project has lost real findings by
    deleting, and the fix for a bad backlog is better evidence, not amnesia."""
    store = _store(tmp_path)
    ids = _propose(store, 2)
    _count(tmp_path)

    assert all(store.get(hid).status is HypothesisStatus.PROPOSED for hid in ids)


def test_a_naive_timestamp_does_not_silently_disable_the_filter(tmp_path, caplog):
    """The bug this filter shipped with, and the reason it is worth a test.

    Campaign stamps come back timezone-aware; `Hypothesis.created_at` is naive.
    Comparing them raised `TypeError`, a blanket `except Exception` swallowed
    it, and the filter became a no-op — 46 rows in, 46 out, on a workspace where
    every one was stale. A broken guard reporting healthy is the failure this
    project keeps paying for.
    """
    from datetime import datetime

    from labpilot.research_engine.execution.technique.vocabulary import campaigns_since

    naive = datetime(2026, 8, 2, 22, 5, 19)
    aware = datetime.fromisoformat("2026-08-07T10:00:00+00:00")

    # Must not raise, and must count the later campaign.
    assert campaigns_since(naive, (aware,)) == 1


def test_the_filter_reports_rather_than_swallows_an_unexpected_shape(monkeypatch, caplog):
    """Narrow except, and logged: the next mismatch must be findable.

    Written first with `created_at=object()` and empty selections, which never
    reached the guard — `_parse_timestamp` returns `None` for an unknown shape
    and `campaigns_since` then returns 0 without raising, so the test passed
    while the branch it names stayed unexecuted. Raising from `campaigns_since`
    directly tests the guard instead of hoping an input still triggers it.
    """
    import logging

    from labpilot.research_engine.intelligence.hypothesis import viability

    def _explode(created_at, stamps):
        raise TypeError("can't compare offset-naive and offset-aware datetimes")

    monkeypatch.setattr(
        "labpilot.research_engine.execution.technique.vocabulary.campaigns_since", _explode
    )

    class _Broken:
        id = "H-001"
        evidence_for = ()
        evidence_against = ()
        created_at = "2026-08-07T10:00:00+00:00"

    with caplog.at_level(logging.WARNING):
        assert viability._is_stale(_Broken(), ((_STAMP, "H-002"),)) is False

    assert "treating as live" in caplog.text


def test_the_count_never_raises_even_when_the_filter_does(tmp_path, monkeypatch, caplog):
    """The docstring promises "never raises" and only the store read was
    guarded. `_is_stale` narrows its own catch to (TypeError, ValueError), so
    anything else — an AttributeError off a malformed row — propagated through
    `should_gather_evidence` into the policy step and ended the campaign: the
    gate stuck shut, reached by crashing instead of by lying."""
    import logging

    from labpilot.research_engine.intelligence.hypothesis import viability

    _propose(_store(tmp_path), 4)
    monkeypatch.setattr(
        viability,
        "_selection_times",
        lambda *a, **k: ((_STAMP, "H-999"),),
    )
    monkeypatch.setattr(
        viability,
        "_is_stale",
        lambda *a, **k: (_ for _ in ()).throw(AttributeError("malformed row")),
    )

    with caplog.at_level(logging.WARNING):
        assert _count(tmp_path) == 4

    assert "counting the whole pool" in caplog.text


# --- a hypothesis is not aged by its own selections --------------------------


def test_its_own_retries_do_not_make_a_hypothesis_stale(tmp_path):
    """`record_failed_attempt` returns a RETRYABLE failure to `proposed` with no
    evidence written, so a hypothesis that was planned once is back in this pool
    — and `_selection_times` counted its own plan against it.

    H-001 selected, failed on a rate limit with two of three attempts unused,
    back to `proposed`. One selection by anyone else and it aged out, thinning
    the pool that decides whether the campaign may look for better ideas.
    """
    store = _store(tmp_path)
    mine, other = _propose(store, 2)
    # My own selection, then one by someone else: two stamps, but only one of
    # them is the selector preferring something over me.
    _select(tmp_path, mine, other)

    assert _count(tmp_path) == 2


def test_being_passed_over_twice_is_still_stale(tmp_path):
    """The filter must keep working — the fix narrows what counts, it does not
    switch the rule off."""
    store = _store(tmp_path)
    ids = _propose(store, 3)
    passed_over = ids[0]
    _select(tmp_path, ids[1], ids[2])

    assert STALE_AFTER_SELECTIONS == 2
    assert _count(tmp_path) == 2
    assert store.get(passed_over).status is HypothesisStatus.PROPOSED


def test_one_pass_over_is_not_yet_stale(tmp_path):
    """Below the threshold nothing is excluded."""
    store = _store(tmp_path)
    ids = _propose(store, 2)
    _select(tmp_path, ids[1])

    assert _count(tmp_path) == 2
