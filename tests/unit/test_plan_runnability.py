"""One definition of "runnable", and one of "unrun".

Three call sites had each written their own version and none matched: the
Engineer accepts {ready, in_progress} (reopening abandoned first), `_latest_plan_id`
used {ready, in_progress, draft}, and `has_unrun_plan` used {ready, draft}. The
Conductor consequently offered `run_plan` for finished plans, the Engineer
refused with "status=done; need ready or in_progress", and the campaign lost a
step each time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    is_runnable_plan_status,
    is_unrun_plan_status,
)
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

_COMP = "demo"


class _WS:
    """Workspace facade with a real knowledge dir, so the query really runs."""

    def __init__(self, tmp_path):
        self.knowledge_dir = tmp_path / "knowledge"
        self.competition = _COMP
        self.root = tmp_path


def _plan(store, pid, status, hypothesis_id=""):
    now = datetime.now(UTC)
    store.upsert_plan(
        ResearchPlan(
            id=pid,
            competition=_COMP,
            hypothesis_id=hypothesis_id,
            goal="g",
            status=status,
            created_at=now,
            updated_at=now,
        )
    )


@pytest.mark.parametrize(
    ("status", "runnable"),
    [
        (PlanStatus.READY, True),
        (PlanStatus.IN_PROGRESS, True),
        # The Engineer reopens an abandoned plan to READY before checking.
        (PlanStatus.ABANDONED, True),
        # A draft has not been compiled: the Engineer refuses it.
        (PlanStatus.DRAFT, False),
        (PlanStatus.DONE, False),
    ],
)
def test_runnable_matches_what_the_engineer_accepts(status, runnable):
    assert is_runnable_plan_status(status) is runnable
    assert is_runnable_plan_status(str(status)) is runnable


@pytest.mark.parametrize(
    ("status", "unrun"),
    [
        (PlanStatus.DRAFT, True),
        (PlanStatus.READY, True),
        (PlanStatus.IN_PROGRESS, True),
        (PlanStatus.ABANDONED, True),
        (PlanStatus.DONE, False),
    ],
)
def test_unrun_is_wider_than_runnable(status, unrun):
    assert is_unrun_plan_status(status) is unrun


def test_the_two_sets_are_deliberately_different():
    """A draft is outstanding work that cannot be dispatched. Collapsing the
    two questions into one is what produced the bug."""
    assert is_unrun_plan_status(PlanStatus.DRAFT)
    assert not is_runnable_plan_status(PlanStatus.DRAFT)


def test_an_unknown_status_is_neither():
    assert not is_runnable_plan_status("garbage")
    assert not is_unrun_plan_status(None)


# --- the policy gate --------------------------------------------------------


def test_all_plans_done_offers_generate_not_run(tmp_path):
    from labpilot.research_engine.conductor.policy import has_runnable_plan, has_unrun_plan

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.DONE)
    finally:
        store.close()

    assert has_runnable_plan(_WS(tmp_path)) is False
    assert has_unrun_plan(_WS(tmp_path)) is False


def test_a_ready_plan_is_runnable(tmp_path):
    from labpilot.research_engine.conductor.policy import has_runnable_plan

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.READY)
    finally:
        store.close()

    assert has_runnable_plan(_WS(tmp_path)) is True


def test_a_draft_blocks_generating_but_does_not_enable_running(tmp_path):
    from labpilot.research_engine.conductor.policy import has_runnable_plan, has_unrun_plan

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.DRAFT)
    finally:
        store.close()

    assert has_unrun_plan(_WS(tmp_path)) is True
    assert has_runnable_plan(_WS(tmp_path)) is False


def test_a_plan_for_a_retired_hypothesis_is_not_runnable(tmp_path):
    """Retiring an idea must retire the work queued against it.

    Measured on rogii 2026-08-09: `H-051` was correctly rejected and the very
    next step selected `P-021`, the plan carrying it — still `in_progress` and
    therefore still runnable. Answered by a join against the mirrored
    `hypotheses` table, which `HypothesisStore._save` keeps current on every
    mutation.
    """
    from labpilot.research_engine.conductor.loop import _latest_plan_id
    from labpilot.research_engine.conductor.policy import has_runnable_plan

    hstore = HypothesisStore(tmp_path / "knowledge", _COMP)
    hyp = hstore.create(observation="o", reason="r", prediction="p", confidence=0.5)
    hstore.update_status(hyp.id, HypothesisStatus.REJECTED)

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.IN_PROGRESS, hypothesis_id=hyp.id)
    finally:
        store.close()

    assert has_runnable_plan(_WS(tmp_path)) is False
    assert _latest_plan_id(_WS(tmp_path)) is None


def test_a_plan_for_a_live_hypothesis_stays_runnable(tmp_path):
    """The carve-out must not cost the behaviour it guards."""
    from labpilot.research_engine.conductor.policy import has_runnable_plan

    hstore = HypothesisStore(tmp_path / "knowledge", _COMP)
    hyp = hstore.create(observation="o", reason="r", prediction="p", confidence=0.5)

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.READY, hypothesis_id=hyp.id)
    finally:
        store.close()

    assert has_runnable_plan(_WS(tmp_path)) is True


def test_a_baseline_plan_is_runnable_with_no_hypothesis(tmp_path):
    """No hypothesis means no retired idea behind it."""
    from labpilot.research_engine.conductor.policy import has_runnable_plan

    store = PlanStore(tmp_path / "knowledge", _COMP)
    try:
        _plan(store, "P-001", PlanStatus.READY, hypothesis_id="")
    finally:
        store.close()

    assert has_runnable_plan(_WS(tmp_path)) is True
