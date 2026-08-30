"""The two breakers that end a campaign are settable, and default when not set.

Measured on playground-series-s6e8 (2026-08-30): a campaign spent 12 of 16
steps on `implement` — the policy front-loads them while unimplemented
hypotheses remain — and implements are not executions, so `steps_since_success`
climbed to 8 while nothing was wrong. The first ordinary failure, an all-NaN
generated feature the repair loop fixes in one round, then tripped
`max_barren_steps` and ended the run.

The defaults are right for the case they were built from: rogii's S-021 spent
30 steps producing no execution at all, which is why the counter advances every
step rather than per execution. What was missing is any way for an operator who
knows their campaign is implement-heavy to say so. Hence flags, not new
behaviour — an unset flag must still yield exactly the shipped default.
"""

from __future__ import annotations

import pytest

from labpilot.cli.conduct import _budget_metadata
from labpilot.research_engine.conductor.budgets import (
    DEFAULT_MAX_BARREN_STEPS,
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
)

BASE = {
    "max_submissions": None,
    "max_wall_s": None,
    "max_cost_usd": None,
    "target_metric": None,
    "target_value": None,
    "plateau_window": 3,
}


def _budgets(**overrides: object) -> dict:
    return _budget_metadata(**BASE, **overrides)["budgets"]


def test_an_unset_flag_keeps_the_shipped_default() -> None:
    """The whole risk of this change: a flag nobody passed must not disable a
    breaker. `None` means *disabled* in `BudgetConfig`, so passing it straight
    through would silently remove the circuit breaker from every campaign that
    did not mention it.
    """
    budgets = _budgets()

    assert budgets["max_barren_steps"] == DEFAULT_MAX_BARREN_STEPS
    assert budgets["max_consecutive_failures"] == DEFAULT_MAX_CONSECUTIVE_FAILURES


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_barren_steps", 25), ("max_consecutive_failures", 10)],
)
def test_a_set_flag_reaches_the_budget(field: str, value: int) -> None:
    assert _budgets(**{field: value})[field] == value


def test_each_flag_moves_only_itself() -> None:
    """Raising one breaker must not quietly raise the other; they guard
    different failures — a stalled policy versus unrepairable code."""
    budgets = _budgets(max_barren_steps=25)

    assert budgets["max_barren_steps"] == 25
    assert budgets["max_consecutive_failures"] == DEFAULT_MAX_CONSECUTIVE_FAILURES
