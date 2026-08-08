"""The breaker must count what the experiment did, not what the call returned.

A tool call returning is not an experiment succeeding. `run_plan` reports a
failed execution in `data["status"]` and raises nothing at all, so counting a
clean return as a success reset the breaker every time one ran.

Measured on rogii 2026-08-09: a campaign with **8 executions, every one failed**
ran its full 8 steps. Ten of its sixteen dispatches were `run_plan` returning
normally, and each reset `consecutive_failures` to zero — so the breaker built
to stop exactly this never reached 3.

Same shape as the defects M20 collects: the gate tested the easier thing. This
one was committed by the author of the breaker, one day after writing it.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.conductor.loop import _experiment_outcome


class _Result:
    def __init__(self, **data):
        self.data = data


def test_a_failed_execution_is_a_failure_even_though_the_call_returned():
    """The exact rogii case: `run_plan` returns, execution failed."""
    succeeded, error = _experiment_outcome(
        _Result(status="failed", error="task P-021-T04 failed: smoke gate")
    )

    assert succeeded is False
    assert "P-021-T04" in error


def test_a_succeeded_execution_is_a_success():
    assert _experiment_outcome(_Result(status="succeeded")) == (True, "")


@pytest.mark.parametrize("status", ["pending", "running", "cancelled", ""])
def test_an_unfinished_execution_is_not_a_success(status):
    """An execution left mid-flight has produced nothing. Treating it as a win
    would reset the breaker on a campaign that is stalling, not progressing."""
    succeeded, error = _experiment_outcome(_Result(status=status))

    assert succeeded is False
    assert error


def test_a_failure_without_an_error_message_still_says_something():
    """`stop:failing` with no reason reproduces the original complaint."""
    succeeded, error = _experiment_outcome(_Result(status="failed"))

    assert succeeded is False
    assert "failed" in error


def test_a_result_with_no_status_is_treated_as_success():
    """This runs only where the call *worked*. Inventing failures from a
    missing field would stop campaigns for a reporting gap rather than a real
    one — and tools that report no status are not experiment tools."""
    assert _experiment_outcome(_Result(files=3)) == (True, "")


@pytest.mark.parametrize("result", [None, object(), "not a result"])
def test_an_unexpected_result_shape_does_not_stop_the_campaign(result):
    assert _experiment_outcome(result) == (True, "")


def test_status_matching_is_case_and_space_insensitive():
    assert _experiment_outcome(_Result(status=" Succeeded ")) == (True, "")
