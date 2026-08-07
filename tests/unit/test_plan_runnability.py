"""One definition of "runnable", and one of "unrun".

Three call sites had each written their own version and none matched: the
Engineer accepts {ready, in_progress} (reopening abandoned first), `_latest_plan_id`
used {ready, in_progress, draft}, and `has_unrun_plan` used {ready, draft}. The
Conductor consequently offered `run_plan` for finished plans, the Engineer
refused with "status=done; need ready or in_progress", and the campaign lost a
step each time.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    is_runnable_plan_status,
    is_unrun_plan_status,
)


class _Plan:
    def __init__(self, pid: str, status: str) -> None:
        self.id = pid
        self.status = status
        self.metadata: dict = {}


class _WS:
    knowledge_dir = "/nonexistent"
    competition = "demo"


@pytest.fixture
def plans(monkeypatch):
    """Install a fake PlanArtifacts returning whatever the test sets."""
    holder: list[_Plan] = []

    class _Artifacts:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return list(holder)

        def close(self):
            pass

    monkeypatch.setattr("labpilot.research_engine.artifacts.plan.PlanArtifacts", _Artifacts)
    return holder


# --- the predicates ---------------------------------------------------------


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


def test_all_plans_done_offers_generate_not_run(plans):
    """The exact step-burner: every plan finished, `run_plan` still offered."""
    from labpilot.research_engine.conductor.policy import has_runnable_plan, has_unrun_plan

    plans.extend([_Plan("P-001", "done"), _Plan("P-002", "done")])
    assert has_runnable_plan(_WS()) is False
    assert has_unrun_plan(_WS()) is False


def test_a_ready_plan_is_runnable(plans):
    from labpilot.research_engine.conductor.policy import has_runnable_plan

    plans.extend([_Plan("P-001", "done"), _Plan("P-002", "ready")])
    assert has_runnable_plan(_WS()) is True


def test_a_draft_blocks_generating_but_does_not_enable_running(plans):
    from labpilot.research_engine.conductor.policy import has_runnable_plan, has_unrun_plan

    plans.append(_Plan("P-001", "draft"))
    assert has_unrun_plan(_WS()) is True
    assert has_runnable_plan(_WS()) is False


def test_no_plans_at_all(plans):
    from labpilot.research_engine.conductor.policy import has_runnable_plan, has_unrun_plan

    assert has_runnable_plan(_WS()) is False
    assert has_unrun_plan(_WS()) is False


# --- the id resolver --------------------------------------------------------


def test_latest_plan_id_returns_the_newest_runnable(plans):
    import labpilot.research_engine.conductor.loop as loop_mod

    plans.extend([_Plan("P-001", "ready"), _Plan("P-009", "done"), _Plan("P-004", "ready")])
    assert loop_mod._latest_plan_id(_WS()) == "P-004"


def test_latest_plan_id_is_none_when_all_are_done(plans):
    import labpilot.research_engine.conductor.loop as loop_mod

    plans.extend([_Plan("P-001", "done"), _Plan("P-009", "done")])
    assert loop_mod._latest_plan_id(_WS()) is None
