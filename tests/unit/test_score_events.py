"""M8-1: the comparable score series has a durable place to live.

`ScoreEvent` carries what a flat `list[float]` cannot — the experiment id a
later hypothesis has to cite, and the resolved metric key that keeps two
runs comparable. Nothing writes `score_events` yet; M8-2 adds the writer and
the `metric_history`/`last_metric` derivation together, where a real producer
makes both testable.
"""

from __future__ import annotations

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


def test_adding_the_series_leaves_the_existing_metric_fields_alone():
    """M8-1 must not touch `metric_history`/`last_metric`. They are read by
    `evaluate_stops` and written by nothing; M8-2 owns the derivation, so a
    stored value has to survive a load/save round trip untouched until then."""
    stored = {"budget_state": {"metric_history": [0.5, 0.501], "last_metric": 0.501}}

    cfg, state = budgets_from_metadata(stored)

    assert state.metric_history == [0.5, 0.501]
    assert state.last_metric == 0.501
    assert budgets_to_metadata({}, cfg, state)["budget_state"]["metric_history"] == [0.5, 0.501]
