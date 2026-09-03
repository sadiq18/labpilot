"""Three identical failures is a stall; three distinct ones is progress.

Issue #173. A campaign on playground-series-s6e8 (2026-08-30) ran the
implement -> run_experiment cycle and stopped at `stop:failing` with three
failures that were *not* the same failure:

    P-001-T03  pipeline/infer.py imports train, undeclared in its PEP 723 block
    P-002-T04  LGBMClassifier.fit() got an unexpected keyword argument 'verbose'
    P-003-T05  pandas comparison error in arraylike.py

Each attempt fixed the previous defect and surfaced the next — the repair loop
working, interrupted because the breaker counts failures rather than asking
whether they are the same one. `DEFAULT_MAX_CONSECUTIVE_FAILURES`'s own comment
already said "the third *identical* one is a pattern"; only the prose knew.

The distinct case stays bounded by `max_barren_steps`, which is asserted here
too — "no longer stops at three" must not mean "never stops".
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.conductor.budgets import (
    DEFAULT_MAX_CONSECUTIVE_FAILURES,
    BudgetConfig,
    BudgetState,
    evaluate_stops,
)

#: The three failures from the measured run, in the order they surfaced.
CONVERGING = (
    "pipeline/infer.py imports train but its PEP 723 block does not declare it",
    "LGBMClassifier.fit() got an unexpected keyword argument 'verbose'",
    "pandas comparison error in arraylike.py",
)


def _after(*errors: str, config: BudgetConfig | None = None) -> tuple[BudgetConfig, BudgetState]:
    state = BudgetState()
    for error in errors:
        state.record_execution(succeeded=False, error=error)
    return config or BudgetConfig(), state


def test_three_distinct_failures_do_not_stop_the_campaign() -> None:
    """The issue's case: the loop was peeling layers and had one more to go."""
    config, state = _after(*CONVERGING)

    assert state.consecutive_failures == DEFAULT_MAX_CONSECUTIVE_FAILURES
    assert state.failures_are_repeating() is False
    assert evaluate_stops(config, state) == "none"


def test_three_identical_failures_still_stop_it() -> None:
    """The behaviour the default was chosen for, unchanged."""
    stuck = "ModuleNotFoundError: No module named 'catboost'"
    config, state = _after(stuck, stuck, stuck)

    assert state.failures_are_repeating() is True
    assert evaluate_stops(config, state) == "failing"


def test_the_same_failure_is_recognised_through_moving_ids_and_line_numbers() -> None:
    """One defect reported twice is one defect. Ids are per-attempt by
    construction and line numbers move, so leaving them in would make every
    repeat look novel and defeat the check entirely."""
    config, state = _after(
        "P-001-T03 failed at train.py:214: LGBMClassifier.fit() got an "
        "unexpected keyword argument 'verbose'",
        "P-002-T04 failed at train.py:227: LGBMClassifier.fit() got an "
        "unexpected keyword argument 'verbose'",
        "P-003-T05 failed at train.py:231: LGBMClassifier.fit() got an "
        "unexpected keyword argument 'verbose'",
    )

    assert state.failures_are_repeating() is True
    assert evaluate_stops(config, state) == "failing"


def test_a_converging_loop_still_stops_on_barren_steps() -> None:
    """Not unbounded. Distinct failures accrue steps like any other, and the
    slower breaker is what ends a loop that never converges."""
    config = BudgetConfig(max_barren_steps=4)
    _, state = _after(*CONVERGING, config=config)
    assert evaluate_stops(config, state) == "none"

    state.steps_since_success = 4
    assert evaluate_stops(config, state) == "failing"


def test_failures_with_no_text_keep_the_old_behaviour() -> None:
    """A failure that records no error appends nothing, so there is nothing to
    compare. Answering "repeating" there stops the campaign exactly as before
    rather than running on a signal that does not exist."""
    config, state = _after("", "", "")

    assert state.recent_failures == []
    assert state.failures_are_repeating() is True
    assert evaluate_stops(config, state) == "failing"


def test_one_recorded_failure_is_not_evidence_of_novelty() -> None:
    """Below the threshold this changes nothing, but the predicate must not
    claim a campaign is converging on the strength of a single data point."""
    _, state = _after(CONVERGING[0])

    assert state.failures_are_repeating() is True


def test_a_success_clears_the_history() -> None:
    """A campaign that recovers is not judged on the failures it climbed out
    of — including for this predicate."""
    _, state = _after(*CONVERGING)
    state.record_execution(succeeded=True)

    assert state.recent_failures == []
    assert state.consecutive_failures == 0


@pytest.mark.parametrize("raised", [5, 10])
def test_a_raised_threshold_still_reads_a_stall(raised: int) -> None:
    """`recent_failures` is bounded at three, so the predicate cannot look back
    over a longer window — it asks whether the last two agree, which is
    answerable at any threshold."""
    stuck = "ModuleNotFoundError: No module named 'catboost'"
    config = BudgetConfig(max_consecutive_failures=raised)
    _, state = _after(*([stuck] * raised), config=config)

    assert evaluate_stops(config, state) == "failing"


# --- where the limits come from ----------------------------------------------


def test_a_workspace_can_set_the_breakers_in_its_config() -> None:
    """Issue #173's first bullet. The flag existed; the config field did not, so
    an operator whose models need five attempts retyped it every invocation."""
    from labpilot.cli.conduct import _breaker, _configured
    from labpilot.config import AppConfig

    configured = AppConfig.model_validate(
        {"campaign": {"max_consecutive_failures": 6, "max_barren_steps": 20}}
    ).campaign

    assert _configured(None, configured.max_consecutive_failures) == 6
    # A flag still wins over the file.
    assert _configured(2, configured.max_consecutive_failures) == 2
    assert _breaker("max_consecutive_failures", _configured(None, 6)) == {
        "max_consecutive_failures": 6
    }


def test_an_unconfigured_workspace_keeps_the_shipped_default() -> None:
    """`None` at both layers must reach `BudgetConfig` as *absent*, not as the
    `None` that disables a breaker outright."""
    from labpilot.cli.conduct import _breaker, _configured
    from labpilot.config import AppConfig

    campaign = AppConfig().campaign

    assert campaign.max_consecutive_failures is None
    assert _breaker("max_consecutive_failures", _configured(None, None)) == {}


def test_zero_in_the_config_disables_the_breaker() -> None:
    """The reason zero carries the disable: `None` is spent on "not set", so a
    configured `0` is the only way to say "off" from a file."""
    from labpilot.cli.conduct import _breaker, _configured
    from labpilot.config import AppConfig

    campaign = AppConfig.model_validate({"campaign": {"max_barren_steps": 0}}).campaign

    assert _breaker("max_barren_steps", _configured(None, campaign.max_barren_steps)) == {
        "max_barren_steps": None
    }


def test_a_negative_limit_is_refused_at_the_config_boundary() -> None:
    """`-1` would make `consecutive_failures >= -1` true before anything ran."""
    import pydantic

    from labpilot.config import AppConfig

    with pytest.raises(pydantic.ValidationError):
        AppConfig.model_validate({"campaign": {"max_barren_steps": -1}})
