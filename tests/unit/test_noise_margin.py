"""What "beats the floor" has to mean before it means anything.

The plan: *"strictly better in the metric's declared direction by more than the
fold-to-fold std. Not 'better by any epsilon' — that is noise."*

Tier 2 runs this against real datasets where the model clears the floor by six to
twenty-nine times the fold spread — so the *threshold* is never exercised there,
and mutations replacing it with `gap > 0` passed the whole file. These drive the
margin by hand, which is the only place the bar itself is under test.
"""

from __future__ import annotations

from labpilot.research_engine.execution.baseline.baseline_one import (
    ModelReading,
    beats_floor_beyond_noise,
)
from labpilot.research_engine.execution.baseline.floor import FloorReading


def _floor(score: float) -> FloorReading:
    return FloorReading(metric_name="m", score=score, best_strategy="mean", fold_scores=[score] * 5)


def _model(score: float, folds: list[float]) -> ModelReading:
    return ModelReading(metric_name="m", score=score, fold_scores=folds)


def test_a_win_inside_the_fold_spread_is_not_a_win() -> None:
    """The sentence the plan is at pains about. A model whose folds disagree by
    more than its margin over the floor has not shown it is better — it has
    shown the folds disagree."""
    margin = beats_floor_beyond_noise(
        _floor(0.70), _model(0.72, [0.60, 0.80, 0.65, 0.78, 0.77]), "maximize"
    )

    assert margin.gap == 0.02 or abs(margin.gap - 0.02) < 1e-9
    assert margin.noise > margin.gap
    assert not margin.beats_noise
    assert "does not clear" in margin.reason


def test_a_win_outside_the_fold_spread_counts() -> None:
    margin = beats_floor_beyond_noise(
        _floor(0.60), _model(0.82, [0.81, 0.82, 0.83, 0.82, 0.82]), "maximize"
    )

    assert margin.beats_noise and margin.reason == ""


def test_the_direction_decides_which_way_is_better() -> None:
    """An RMSLE of 0.13 beats a floor of 0.40; an accuracy of 0.13 does not."""
    lower_is_better = beats_floor_beyond_noise(
        _floor(0.40), _model(0.13, [0.12, 0.13, 0.14, 0.13, 0.13]), "minimize"
    )
    higher_is_better = beats_floor_beyond_noise(
        _floor(0.40), _model(0.13, [0.12, 0.13, 0.14, 0.13, 0.13]), "maximize"
    )

    assert lower_is_better.beats_noise
    assert not higher_is_better.beats_noise


def test_a_single_fold_has_no_spread_to_measure_against() -> None:
    """One number is not a distribution, and `ddof=1` over it is undefined.

    Reporting `beats_noise` here would be answering a question nothing was asked
    — the comparison the plan describes does not exist without folds.
    """
    margin = beats_floor_beyond_noise(_floor(0.60), _model(0.99, [0.99]), "maximize")

    assert not margin.beats_noise
    assert "fewer than two folds" in margin.reason


def test_an_undefined_reading_is_not_a_comparison() -> None:
    undefined = ModelReading(metric_name="m", undefined_reason="cannot run here")

    assert not beats_floor_beyond_noise(_floor(0.6), undefined, "maximize").beats_noise
    assert not beats_floor_beyond_noise(_floor(0.6), None, "maximize").beats_noise


def test_an_unknown_direction_refuses_rather_than_guessing() -> None:
    """A margin computed with the wrong sign is the most convincing wrong number
    this comparison could produce."""
    margin = beats_floor_beyond_noise(
        _floor(0.60), _model(0.82, [0.81, 0.82, 0.83, 0.82, 0.82]), ""
    )

    assert not margin.beats_noise
    assert "maximize" in margin.reason


def test_matching_the_floor_exactly_is_not_beating_it() -> None:
    margin = beats_floor_beyond_noise(_floor(0.70), _model(0.70, [0.70] * 5), "maximize")

    assert margin.gap == 0.0
    assert not margin.beats_noise
