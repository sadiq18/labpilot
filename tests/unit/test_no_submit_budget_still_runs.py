"""`max_submissions=0` means do not submit, not do not run.

Measured 2026-08-08: a campaign started with `--max-submissions 0`, intending
"run but never upload", stopped before its first step with one `stop` decision
and no research done. `evaluate_stops` fired on `0 >= 0` — a cap being checked
before the campaign had a chance to spend it.

The second half matters as much: enforcement belongs in the allowlist, not the
approval gate. `--yes` maps every gated tool to `auto_approve`, so in a
non-interactive run approval is not a brake — a campaign told not to submit
could still upload because nobody was at the terminal to decline.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.conductor.approvals import (
    SUBMIT_TOOLS,
    auto_approve,
    gated_tools_for_autonomy,
)
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    evaluate_stops,
    submit_tools_allowed,
)


def test_a_zero_submission_budget_does_not_stop_a_campaign_that_has_done_nothing():
    stop = evaluate_stops(BudgetConfig(max_submissions=0), BudgetState(submissions=0))
    assert stop == "none"


def test_a_zero_submission_budget_removes_the_submit_tools():
    assert submit_tools_allowed(BudgetConfig(max_submissions=0)) is False
    allowlist = {"generate_plan", "run_experiment", "submit", "submit_learn"}
    assert allowlist - SUBMIT_TOOLS == {"generate_plan", "run_experiment"}


@pytest.mark.parametrize("budget", [None, 1, 5])
def test_a_real_budget_still_offers_the_tools(budget):
    assert submit_tools_allowed(BudgetConfig(max_submissions=budget)) is True


def test_a_spent_budget_still_stops():
    """The cap must keep working — this is a carve-out for zero, not a removal."""
    stop = evaluate_stops(BudgetConfig(max_submissions=1), BudgetState(submissions=1))
    assert stop == "submission_budget"


def test_an_unspent_budget_does_not_stop():
    stop = evaluate_stops(BudgetConfig(max_submissions=2), BudgetState(submissions=1))
    assert stop == "none"


def test_approval_is_not_a_brake_in_a_non_interactive_run():
    """Why enforcement is in the allowlist rather than the gate.

    `submit_learn` is gated at every autonomy level, and `--yes` approves it
    anyway. Pinning this so a future change cannot quietly make the gate the
    only thing standing between a campaign and a Kaggle upload.
    """
    assert "submit_learn" in gated_tools_for_autonomy(0)
    assert "submit_learn" in gated_tools_for_autonomy(1)
    assert auto_approve("submit_learn").decision == "approve"
