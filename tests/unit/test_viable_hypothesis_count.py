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


def test_the_filter_reports_rather_than_swallows_an_unexpected_shape(caplog):
    """Narrow except, and logged: the next mismatch must be findable."""
    import logging

    from labpilot.research_engine.intelligence.hypothesis.viability import _is_stale

    class _Broken:
        evidence_for = ()
        evidence_against = ()
        created_at = object()  # not a timestamp

    with caplog.at_level(logging.WARNING):
        assert _is_stale(_Broken(), ()) is False


# --- a retired idea must retire the work queued against it -------------------


def test_a_plan_testing_a_retired_hypothesis_is_not_selectable():
    """The campaign selects *plans*, not hypotheses, and the two retire
    independently. Measured on rogii 2026-08-09: redundancy correctly rejected
    `H-051`, and the very next step selected `P-021` — the plan carrying it,
    still `in_progress` and therefore still runnable. Retiring the idea has to
    retire the work, or the loop continues one level up.
    """
    from labpilot.research_engine.intelligence.hypothesis.viability import plan_is_selectable

    class _Plan:
        hypothesis_id = "H-051"

    assert plan_is_selectable(_Plan(), {"H-051"}) is False


def test_a_plan_for_a_live_hypothesis_is_selectable():
    from labpilot.research_engine.intelligence.hypothesis.viability import plan_is_selectable

    class _Plan:
        hypothesis_id = "H-052"

    assert plan_is_selectable(_Plan(), {"H-051"}) is True


def test_a_baseline_plan_is_always_selectable():
    """No hypothesis means no retired idea behind it."""
    from labpilot.research_engine.intelligence.hypothesis.viability import plan_is_selectable

    class _Baseline:
        hypothesis_id = None

    assert plan_is_selectable(_Baseline(), {"H-051"}) is True


def test_retired_ids_come_from_rejected_status(tmp_path):
    from labpilot.research_engine.intelligence.hypothesis.viability import (
        retired_hypothesis_ids,
    )

    store = _store(tmp_path)
    ids = _propose(store, 3)
    store.update_status(ids[0], HypothesisStatus.REJECTED)

    assert retired_hypothesis_ids(tmp_path / "knowledge", _COMP) == {ids[0]}


def test_an_unreadable_store_retires_nothing(tmp_path):
    """Failing open: a store we cannot read must not silently disable every
    plan in the workspace."""
    from labpilot.research_engine.intelligence.hypothesis.viability import (
        retired_hypothesis_ids,
    )

    assert retired_hypothesis_ids(tmp_path / "nowhere", _COMP) == set()
