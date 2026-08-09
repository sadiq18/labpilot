"""The harness, proved against defects that actually shipped.

Each test here is a bug we paid a full campaign to find — twenty to sixty
minutes of wall clock, one bug per run. If a scenario cannot be made to fail by
reverting its fix, it is not evidence that the harness works, so each one names
the fix it is holding down.

Deliberately not covered, because they need a real model or real network: the
paper analyzer's runtime, aider declining an edit that was already applied, a
model returning a blank enum. Those stay acceptance runs.
"""

from __future__ import annotations

import pytest
from helpers.campaign_harness import (
    ScriptedPolicy,
    fails,
    harness,
    ok,
    silent_success,
    writes,
)

from labpilot.research_engine.planner.schemas.task_types import PlanStatus
from labpilot.research_engine.shared.experiments.models import HypothesisStatus


@pytest.fixture
def camp(tmp_path):
    h = harness(tmp_path)
    yield h
    h.close()


# -- the harness can express the shape at all ---------------------------------


def test_a_tool_can_report_success_while_changing_nothing(camp):
    """The whole point. `_echo` cannot express this, which is why the existing
    campaign tests could not catch any of what follows."""
    camp.register("implement", [silent_success()])
    camp.seed_plan(status=PlanStatus.READY)

    trace = camp.run(policy=["implement", None], max_steps=2)

    assert trace.calls("implement") == 1
    assert not (camp.workspace.root / "pipeline" / "train.py").exists()


def test_a_tool_can_change_something(camp):
    """The other half — a harness that can only fail proves nothing either."""
    camp.register("implement", [writes("pipeline/train.py", "print('hi')")])
    camp.seed_plan(status=PlanStatus.READY)

    camp.run(policy=["implement", None], max_steps=2)

    assert (camp.workspace.root / "pipeline" / "train.py").read_text() == "print('hi')"


# -- gating, against seeded domain state --------------------------------------


def test_implement_is_closed_with_no_runnable_plan(camp):
    """Holds down 4415d19. Ungated, `implement` was the one door left open when
    every other closed, and a campaign spent all eight steps there."""
    camp.register("implement", [silent_success()])
    camp.register("run_plan", [ok(status="succeeded")])

    offered = camp.available_tools()

    assert "implement" not in offered
    assert "run_plan" not in offered


def test_implement_opens_with_a_runnable_plan(camp):
    camp.register("implement", [silent_success()])
    camp.seed_plan(status=PlanStatus.READY)

    assert "implement" in camp.available_tools()


def test_a_plan_whose_hypothesis_was_rejected_is_not_selectable(camp):
    """Holds down 28d1cce. Retiring an idea has to retire the work queued
    against it: on rogii `H-051` was rejected and the very next step chose
    `P-021` again."""
    from labpilot.research_engine.planner.store import PlanStore

    rejected = camp.seed_hypothesis(status=HypothesisStatus.REJECTED)
    live = camp.seed_hypothesis(status=HypothesisStatus.PROPOSED)
    camp.seed_plan("P-001", hypothesis_id=rejected, status=PlanStatus.READY)
    camp.seed_plan("P-002", hypothesis_id=live, status=PlanStatus.READY)

    store = PlanStore(camp.workspace.knowledge_dir, camp.workspace.competition)
    try:
        selectable = store.selectable_plan_ids()
    finally:
        store.close()

    assert selectable == ["P-002"]


def test_a_baseline_plan_with_no_hypothesis_stays_selectable(camp):
    """The join must fail open — not knowing must never disable every plan."""
    from labpilot.research_engine.planner.store import PlanStore

    camp.seed_plan("P-001", hypothesis_id="", status=PlanStatus.READY)

    store = PlanStore(camp.workspace.knowledge_dir, camp.workspace.competition)
    try:
        assert store.selectable_plan_ids() == ["P-001"]
    finally:
        store.close()


# -- outcome bookkeeping ------------------------------------------------------


def test_a_raising_tool_never_reports_completed(camp):
    """The floor: whatever else happens, a tool that blew up must not be
    recorded as having worked."""
    camp.register("run_experiment", [fails("no metrics")])
    camp.seed_plan(status=PlanStatus.READY)

    trace = camp.run(policy=["run_experiment", None], max_steps=2)

    assert trace.calls("run_experiment") == 1
    assert "completed" not in trace.task_statuses("run_experiment")


@pytest.mark.xfail(
    reason=(
        "Found by this harness, 2026-08-09. `Scheduler.dispatch` marks a "
        "raising task `retry` while retries remain; `update_task_status` turns "
        "`retry` into `pending`. Nothing in src/ ever re-dispatches a pending "
        "task — `dispatch_next`/`next_ready` have no callers, and the campaign "
        "loop dispatches the task it just enqueued. So every failed task is "
        "parked as `pending` for good, and the observe bundle shows the policy "
        "a growing list of errored tasks that read as queued work."
    ),
    strict=True,
)
def test_a_raising_tool_is_recorded_as_failed(camp):
    camp.register("run_experiment", [fails("no metrics")])
    camp.seed_plan(status=PlanStatus.READY)

    trace = camp.run(policy=["run_experiment", None], max_steps=2)

    assert "failed" in trace.task_statuses("run_experiment")


def test_a_reported_failure_without_an_exception_still_counts(camp):
    """Holds down 240c06a. `run_plan` reports a failed execution in
    `data["status"]` and raises nothing at all, so counting exceptions asked a
    question the breaker was not supposed to ask — eight consecutive failed
    executions all recorded `completed`."""
    from labpilot.research_engine.conductor.loop import _experiment_outcome
    from labpilot.research_engine.tools.descriptors import ToolResult

    succeeded, error = _experiment_outcome(ToolResult(refs=[], data={"status": "failed"}))

    assert succeeded is False
    assert "failed" in error


# -- the policy path, which prefer_offline never reaches ----------------------


def test_the_policy_is_shown_only_available_tools(camp):
    """The allowlist in the prompt is the campaign's real safety boundary."""
    camp.register("implement", [silent_success()])
    camp.register("query_memory", [ok()])
    policy = ScriptedPolicy(["query_memory", None])

    camp.run(policy=policy, max_steps=2)

    assert "implement" not in policy.offered(0)
    assert "query_memory" in policy.offered(0)


def test_choosing_a_gated_tool_does_not_end_the_campaign(camp):
    """Measured 2026-08-07: the policy chose a gated `generate_plan` and the run
    stopped at step 4 with "rejected non-catalog tool". A recoverable policy
    mistake must route into the retry, not kill the session."""
    camp.register("implement", [silent_success()])
    camp.register("query_memory", [ok()])

    trace = camp.run(policy=["implement", "query_memory", None], max_steps=3)

    assert trace.calls("implement") == 0
    assert trace.calls("query_memory") == 1


# -- evidence gathering, without waiting for wall clock -----------------------


def test_fresh_evidence_holds_the_re_sweep_floor(camp):
    """Holds down the floor that made us idle twenty minutes on 2026-08-09.
    Reachable here in milliseconds by moving the data, not the clock."""
    from labpilot.research_engine.conductor.policy import should_gather_evidence

    camp.seed_artifact()

    gather_ok, reason = should_gather_evidence(camp.workspace)

    assert gather_ok is False
    assert "ago" in reason


def test_stale_evidence_reopens_gathering_even_with_a_full_pool(camp):
    """Holds down 660ebc8. The clauses are independent: a queue of stale ideas
    is the strongest reason to look for better ones, not a reason to stop.

    The full pool is the whole test. Written without it, `viable < 5` returns
    True first and the staleness clause never runs — the assertion passes
    against the AND version it is supposed to rule out. Caught by reverting the
    fix and watching this stay green, which is why every scenario here is held
    to that.
    """
    from labpilot.research_engine.conductor.policy import (
        _VIABLE_TARGET,
        should_gather_evidence,
    )

    for _ in range(_VIABLE_TARGET * 2):
        camp.seed_hypothesis()
    camp.seed_artifact()
    camp.age_artifacts(hours=48)

    gather_ok, reason = should_gather_evidence(camp.workspace)

    assert gather_ok is True
    assert "old" in reason, f"reopened for the wrong reason: {reason}"


def test_the_floor_closes_analyze_but_not_the_rest(camp):
    """A rate limit on gathering must not be a rate limit on working."""
    camp.register("analyze_competition", [ok()])
    camp.register("query_memory", [ok()])
    camp.seed_artifact()

    offered = camp.available_tools()

    assert "analyze_competition" not in offered
    assert "query_memory" in offered
