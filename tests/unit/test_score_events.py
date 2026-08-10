"""M8-1: the comparable score series has a durable place to live.

`ScoreEvent` carries what a flat `list[float]` cannot — the experiment id a
later hypothesis has to cite, and the resolved metric key that keeps two
runs comparable. Nothing writes `score_events` yet; M8-2 adds the writer and
the `metric_history`/`last_metric` derivation together, where a real producer
makes both testable.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    budgets_from_metadata,
    budgets_to_metadata,
)


def _event(experiment_id: str, value: float, **overrides: object) -> ScoreEvent:
    defaults: dict[str, object] = {
        "experiment_id": experiment_id,
        "hypothesis_id": "H-001",
        "technique": "target_encoding",
        "metric_name": "cv_rmse",
        "value": value,
        "maximize": False,
        "timestamp": "2026-08-11T00:00:00+00:00",
    }
    defaults.update(overrides)
    return ScoreEvent(**defaults)


def test_a_score_event_carries_what_a_flat_float_list_cannot():
    """The whole reason for the model: a float can't cite an experiment."""
    event = _event("E-042", 190.97)

    assert event.experiment_id == "E-042"
    assert event.hypothesis_id == "H-001"
    assert event.metric_name == "cv_rmse"
    assert event.maximize is False


def test_score_events_survive_the_session_metadata_round_trip():
    """Nested models, not just floats, must come back out of the JSON blob."""
    state = BudgetState(
        score_events=[
            _event("E-001", 194.80),
            _event(
                "E-002",
                190.97,
                technique=None,
                combo_techniques=["mixup", "cutout"],
                hypothesis_id="H-002",
            ),
        ]
    )

    meta = budgets_to_metadata({}, BudgetConfig(), state)
    _, restored = budgets_from_metadata(meta)

    assert [e.experiment_id for e in restored.score_events] == ["E-001", "E-002"]
    assert restored.score_events[0].value == 194.80
    assert restored.score_events[1].combo_techniques == ["mixup", "cutout"]
    assert restored.score_events[1].technique is None
    assert restored.score_events[1].hypothesis_id == "H-002"


def test_the_new_fields_default_empty_and_round_trip():
    """A session written before these fields existed must still load, and a
    session carrying them must not lose them on the next write."""
    _, from_legacy = budgets_from_metadata({"budget_state": {"submissions": 3}})
    assert from_legacy.score_events == []
    assert from_legacy.stagnation_mint_fired is False
    assert from_legacy.submissions == 3

    meta = budgets_to_metadata({}, BudgetConfig(), BudgetState(stagnation_mint_fired=True))
    _, restored = budgets_from_metadata(meta)
    assert restored.stagnation_mint_fired is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_refused(bad):
    """A diverged run's NaN is not a comparable score.

    Two things go wrong if it gets in. It serializes as a bare `NaN` token —
    invalid JSON — into the session blob. And every NaN comparison is False,
    so `evaluate_stops` returns "none" for a fully-plateaued all-NaN window
    and for a NaN `last_metric`: both objective stops go silently dead on a
    campaign whose runs all diverge. The executions *succeeded*, so the
    failure breaker does not cover it either.
    """
    with pytest.raises(ValidationError):
        _event("E-001", bad)


@pytest.mark.parametrize("good", [0.0, -3.2, 194.80, 1e308])
def test_ordinary_scores_still_pass(good):
    """The guard must not cost the values it exists to protect — zero and
    negative metrics are ordinary, not sentinels."""
    assert _event("E-001", good).value == good


def test_the_timestamp_defaults_to_the_package_format():
    """Every other timestamp in this package auto-populates one UTC-aware
    format. A caller left to hand-build one supplies a naive
    `datetime.now().isoformat()` sooner or later, and two events then fail to
    compare with `TypeError: can't compare offset-naive and offset-aware`."""
    first = ScoreEvent(experiment_id="E-001", metric_name="cv_rmse", value=1.0, maximize=False)
    second = ScoreEvent(experiment_id="E-002", metric_name="cv_rmse", value=2.0, maximize=False)

    assert datetime.fromisoformat(first.timestamp).tzinfo is not None
    assert datetime.fromisoformat(first.timestamp) <= datetime.fromisoformat(second.timestamp)


def test_adding_the_series_leaves_the_existing_metric_fields_alone():
    """M8-1 must not touch `metric_history`/`last_metric`. They are read by
    `evaluate_stops` and written by nothing; M8-2 owns the derivation, so a
    stored value has to survive a load/save round trip untouched until then."""
    stored = {"budget_state": {"metric_history": [0.5, 0.501], "last_metric": 0.501}}

    cfg, state = budgets_from_metadata(stored)

    assert state.metric_history == [0.5, 0.501]
    assert state.last_metric == 0.501
    assert budgets_to_metadata({}, cfg, state)["budget_state"]["metric_history"] == [0.5, 0.501]
