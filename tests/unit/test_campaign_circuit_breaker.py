"""A campaign that is failing must stop, and say so.

Every stop condition before this one answered *"am I finished?"* — submissions,
wall time, cost, metric target, plateau, operator pause, step count. None
answered *"is any of this working?"*.

Measured on rogii 2026-08-08: **108 consecutive failed executions**, each ~33 ms,
each the identical `ModuleNotFoundError`, across four sessions and 80 conductor
steps. Nothing stopped. `tasks_failed` read 0 throughout, so the campaign looked
healthy the whole time.

The two counters are deliberately different questions. `consecutive_failures`
catches a campaign whose executions fail; `steps_since_success` catches one that
never reaches an execution at all — rogii's S-021 spent 30 steps choosing
`run_experiment` while `plateau` could not fire (it needs metrics) and
`metric_target` could not fire (it needs a metric).
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.conductor.budgets import (
    DEFAULT_MAX_BARREN_STEPS,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    BudgetConfig,
    BudgetState,
    evaluate_stops,
)

_CATBOOST = "task P-019-T04 failed: ModuleNotFoundError: No module named 'catboost'"


def _failed(n: int, error: str = _CATBOOST) -> BudgetState:
    state = BudgetState()
    for _ in range(n):
        state.record_execution(succeeded=False, error=error)
    return state


def test_the_rogii_loop_stops_instead_of_running_108_times():
    assert evaluate_stops(BudgetConfig(), _failed(108)) == "failing"


def test_it_stops_at_the_threshold_not_later():
    at = _failed(DEFAULT_MAX_CONSECUTIVE_FAILURES)
    below = _failed(DEFAULT_MAX_CONSECUTIVE_FAILURES - 1)

    assert evaluate_stops(BudgetConfig(), at) == "failing"
    assert evaluate_stops(BudgetConfig(), below) == "none"


def test_a_transient_failure_is_forgiven_by_a_success():
    """A campaign that recovers must not be punished for what it climbed out of."""
    state = _failed(DEFAULT_MAX_CONSECUTIVE_FAILURES - 1)
    state.record_execution(succeeded=True)

    assert state.consecutive_failures == 0
    assert evaluate_stops(BudgetConfig(), state) == "none"


def test_a_campaign_that_never_reaches_an_execution_still_stops():
    """S-021 burned 30 steps with nothing to count. No per-execution counter
    would have noticed, because no execution ever ran."""
    state = BudgetState(steps_since_success=DEFAULT_MAX_BARREN_STEPS)

    assert evaluate_stops(BudgetConfig(), state) == "failing"


def test_the_breaker_outranks_the_completion_reasons():
    """A campaign whose every execution fails is not *finished*, and reporting
    it as `plateau` or `max_steps` would read as a normal end."""
    state = _failed(DEFAULT_MAX_CONSECUTIVE_FAILURES)
    state.submissions = 5

    assert evaluate_stops(BudgetConfig(max_submissions=1), state) == "failing"


def test_the_stop_carries_what_broke():
    """A bare `stop:failing` reproduces the original complaint — a campaign that
    ended without saying why."""
    state = _failed(DEFAULT_MAX_CONSECUTIVE_FAILURES)

    assert state.recent_failures
    assert "catboost" in state.recent_failures[-1]


def test_the_failure_record_is_bounded():
    """This is a stop reason, not a log, and it lands in session metadata."""
    state = _failed(50)

    assert len(state.recent_failures) <= 3
    assert all(len(f) <= 200 for f in state.recent_failures)


def test_whitespace_in_a_traceback_is_collapsed():
    state = _failed(1, error="line one\n    line two\n\n  line three")

    assert state.recent_failures[-1] == "line one line two line three"


@pytest.mark.parametrize(
    ("disabled", "state"),
    [
        ("max_consecutive_failures", _failed(200)),
        ("max_barren_steps", BudgetState(steps_since_success=200)),
    ],
)
def test_each_breaker_can_be_turned_off_independently(disabled, state):
    """A campaign deliberately probing a broken workspace is legitimate.

    Each trigger is disabled against the state only *it* would fire on, so a
    passing test cannot be explained by the other trigger being inactive.
    """
    assert evaluate_stops(BudgetConfig(**{disabled: None}), state) == "none"


@pytest.mark.parametrize(
    ("disabled", "state"),
    [
        ("max_barren_steps", _failed(200)),
        ("max_consecutive_failures", BudgetState(steps_since_success=200)),
    ],
)
def test_disabling_one_breaker_leaves_the_other_armed(disabled, state):
    assert evaluate_stops(BudgetConfig(**{disabled: None}), state) == "failing"


def test_a_healthy_campaign_is_untouched():
    """The carve-out must not cost the behaviour it guards."""
    state = BudgetState(steps_since_success=1)
    state.record_execution(succeeded=True)

    assert evaluate_stops(BudgetConfig(), state) == "none"
