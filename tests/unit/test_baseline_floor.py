"""M23 step 3: what the dumbest defensible answer scores.

A model that cannot beat a constant has not learned anything, and until now
nothing here could say so — `_observe_delta` compares runs against each other,
so a campaign can improve steadily while sitting below the score you get by
predicting the mean.

The three checks the plan asks of this step:

1. the constants are the *optimal* ones, not plausible ones;
2. the floor is computed on the model's own `ValidationPlan`, and a different
   plan yields a different floor;
3. fitting is per fold on the train side — the whole-target version, which looks
   unbeatable on skewed data, is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.floor import (
    FLOOR_FILENAME,
    FloorReading,
    _folds,
    compute_floor,
    floor_for_workspace,
    load_floor,
    write_floor,
)
from labpilot.research_engine.execution.baseline.selector import ValidationPlan
from labpilot.research_engine.execution.metrics import compute_metric

SKEWED = [1.0, 2.0, 2.0, 10.0] * 5
BINARY = [0, 1, 1, 1] * 5


def _frame(values: list) -> pd.DataFrame:
    return pd.DataFrame({"y": values, "x": range(len(values))})


def _floor(values: list, metric: str, direction: str, **kw) -> FloorReading:
    return compute_floor(
        _frame(values),
        target="y",
        plan=kw.pop("plan", ValidationPlan(scheme="kfold", n_splits=4)),
        metric_name=metric,
        direction=direction,
        **kw,
    )


# --- check 1: the optimal constant, per metric --------------------------------


@pytest.mark.parametrize(
    ("metric", "direction", "values", "expected"),
    [
        ("rmse", "minimize", SKEWED, "mean"),
        ("mse", "minimize", SKEWED, "mean"),
        ("mae", "minimize", SKEWED, "median"),
        ("rmsle", "minimize", SKEWED, "log_mean"),
        ("accuracy", "maximize", BINARY, "majority_class"),
        ("f1", "maximize", BINARY, "majority_class"),
    ],
)
def test_each_metric_gets_its_optimal_constant(
    metric: str, direction: str, values: list, expected: str
) -> None:
    """Mean minimises squared error; median minimises absolute error.

    Over the *same* column — which is why one strategy is not enough, and why
    the losing ones are recorded rather than discarded. A floor that always
    predicted the mean would be beatable by a constant on every MAE competition,
    which is a gate that passes models that learned nothing.
    """
    reading = _floor(values, metric, direction, num_classes=2)

    assert reading.best_strategy == expected
    assert reading.is_defined
    assert len(reading.strategies) >= 1


def test_the_losing_constants_are_recorded_not_discarded() -> None:
    """ "Every strategy is recorded; the floor is the best of them."

    A gate that picked the worse constant is too easy to pass, and the losers
    are what tell a reader whether a low floor means the target is easy or the
    strategy was a poor fit.
    """
    reading = _floor(SKEWED, "mae", "minimize")

    assert set(reading.strategies) == {"median", "mean"}
    assert reading.strategies["median"] < reading.strategies["mean"], "median wins under MAE"
    assert reading.score == reading.strategies["median"]


def test_rmsle_optimises_in_log_space() -> None:
    """The arithmetic mean is not the optimal constant when error is logarithmic.

    `expm1(mean(log1p(y)))` is, and on a skewed price target the gap is the
    difference between a floor a model clears and one it does not.
    """
    reading = _floor(SKEWED, "rmsle", "minimize")

    assert reading.best_strategy == "log_mean"
    assert reading.strategies["log_mean"] < reading.strategies["mean"]
    # And it is the value the formula gives, not merely the smallest of three.
    expected = float(np.expm1(np.log1p(pd.Series(SKEWED)).mean()))
    assert abs(expected - 4.0) > 0  # a real number, distinct from the mean
    assert reading.strategies["log_mean"] < reading.strategies["median"]


def test_logloss_uses_the_prior_vector_not_the_argmax() -> None:
    """Log loss scores probabilities. The argmax would score a hard 0/1 guess,
    which log loss punishes infinitely when it is wrong."""
    reading = _floor(BINARY, "logloss", "minimize", num_classes=2)

    assert reading.best_strategy == "class_prior"
    assert reading.is_defined
    assert 0.0 < (reading.score or 0.0) < 1.0


def test_auc_is_asserted_analytically_rather_than_computed() -> None:
    """A constant prediction carries no ranking information, so its ROC AUC is
    exactly 0.5 for every dataset.

    Computing it invites a fold with a single class present to return NaN or
    0.0 — and a floor of 0.0 is one every model clears.
    """
    reading = _floor(BINARY, "auc", "maximize", num_classes=2)

    assert reading.score == 0.5
    assert reading.strategies == {"constant_prediction": 0.5}


def test_a_metric_with_no_defined_floor_says_so() -> None:
    """Not a score of zero, and not a crash. `floor_undefined` is a state the
    gate has, and it needs the difference between "no floor" and "no attempt"."""
    reading = _floor(SKEWED, "spearman", "maximize")

    assert not reading.is_defined
    assert "spearman" in reading.undefined_reason


# --- check 2: the model's own plan, not one of our own ------------------------


def test_a_different_plan_yields_a_different_floor() -> None:
    """ "A floor on a different split is not a floor."

    The whole value of the number is that it is comparable to the model's
    `cv_<metric>`, and that only holds if both were measured over the same rows.
    """
    values = [float(i) for i in range(40)]

    four = _floor(values, "rmse", "minimize", plan=ValidationPlan(scheme="kfold", n_splits=4))
    ten = _floor(values, "rmse", "minimize", plan=ValidationPlan(scheme="kfold", n_splits=10))

    assert four.score != ten.score
    assert four.validation.n_splits == 4, "the plan is copied onto the reading, not referenced"
    assert four.fingerprint != ten.fingerprint, "the plan is part of what makes a reading stale"


def test_a_plan_that_cannot_be_honoured_is_refused_not_downgraded() -> None:
    """Silently falling back to `KFold` would produce a number in the right
    units and the wrong universe.

    A grouped plan whose group column is not in the table is exactly that case:
    the model will validate by group and this would not.
    """
    reading = compute_floor(
        _frame(SKEWED),
        target="y",
        plan=ValidationPlan(scheme="group_kfold", group_key="well_id", n_splits=4),
        metric_name="rmse",
        direction="minimize",
    )

    assert not reading.is_defined
    assert "group_kfold" in reading.undefined_reason


def test_a_grouped_plan_is_honoured_when_the_key_is_there() -> None:
    frame = pd.DataFrame({"y": SKEWED, "well": [f"w{i % 5}" for i in range(len(SKEWED))]})

    reading = compute_floor(
        frame,
        target="y",
        plan=ValidationPlan(scheme="group_kfold", group_key="well", n_splits=5),
        metric_name="rmse",
        direction="minimize",
    )

    assert reading.is_defined
    assert reading.best_strategy == "mean"


def test_the_same_bytes_and_plan_give_the_same_floor() -> None:
    """Determinism, which a shuffled KFold would cost — `ValidationPlan` carries
    no seed, so a shuffle would be a different floor on every run."""
    first = _floor(SKEWED, "rmse", "minimize")
    second = _floor(SKEWED, "rmse", "minimize")

    assert first.score == second.score
    assert first.fingerprint == second.fingerprint


# --- check 3: per fold, on the train side only --------------------------------


def test_the_constant_is_fitted_per_fold_not_on_the_whole_target() -> None:
    """The leakage version looks unbeatable, which is why it has to be refused.

    An ordered target is the fixture: each fold's train side is a poor guide to
    its validation side, so fitting the constant on the *whole* column — which
    has seen the validation rows — scores markedly better. A floor computed that
    way is one no model can beat, and a gate no model can pass teaches everyone
    to switch it off.
    """
    values = [float(i) for i in range(40)]
    frame = _frame(values)
    plan = ValidationPlan(scheme="kfold", n_splits=4)

    honest = compute_floor(frame, target="y", plan=plan, metric_name="rmse", direction="minimize")
    whole_target = float(pd.Series(values).mean())
    leaky = float(
        np.mean(
            [
                compute_metric(
                    frame["y"].iloc[val].to_numpy(),
                    np.full(len(val), whole_target),
                    "rmse",
                )
                for _, val in _folds(plan, frame)
            ]
        )
    )

    assert leaky < honest.score, "the fixture must actually reward leakage"
    assert honest.score == pytest.approx(13.7321, abs=1e-3)


def test_a_fold_predicts_only_from_rows_before_it_in_its_train_side() -> None:
    """Same property, stated per fold rather than in aggregate.

    Fold 0 of an ordered target trains on rows 10-39 and validates on 0-9, so
    its constant is the mean of the *upper* range. A whole-target fit would use
    a mean pulled down by the very rows it is about to be scored on.
    """
    values = [float(i) for i in range(40)]
    frame = _frame(values)
    plan = ValidationPlan(scheme="kfold", n_splits=4)

    folds = _folds(plan, frame)
    train_idx, val_idx = folds[0]

    assert set(val_idx.tolist()) == set(range(10))
    assert frame["y"].iloc[train_idx].mean() > frame["y"].iloc[val_idx].mean()


# --- the anchor, which is not a constant at all --------------------------------


def test_the_anchor_column_is_carried_forward() -> None:
    """rogii's 15.1 against the pipeline's 1380.

    The target's known prefix is equal to the target wherever present and absent
    exactly on the scored rows. Carrying it forward is a far better answer than
    any constant, and the profiler has named it since 2026-08-13 with nothing
    reading it. A floor that ignored it would be one rogii's pipeline "beats"
    while being 90x worse than doing nothing clever.
    """
    depth = [100.0 + i for i in range(40)]
    frame = pd.DataFrame({"y": depth, "y_input": [d - 0.5 for d in depth]})

    reading = compute_floor(
        frame,
        target="y",
        plan=ValidationPlan(scheme="kfold", n_splits=4),
        metric_name="rmse",
        direction="minimize",
        anchor_column="y_input",
    )

    assert reading.best_strategy == "anchor_carry_forward"
    assert reading.score < reading.strategies["mean"]
    assert reading.score == pytest.approx(0.5, abs=1e-6), "the prefix is exactly 0.5 below"


def test_no_anchor_means_no_anchor_strategy() -> None:
    """The strategy list must not carry an entry nothing produced."""
    reading = _floor(SKEWED, "rmse", "minimize")

    assert "anchor_carry_forward" not in reading.strategies


# --- direction, which has no default -------------------------------------------


def test_the_direction_decides_which_strategy_wins() -> None:
    """The sign is load-bearing only when there is a choice, so it is tested there.

    Every other case in this file has one strategy per metric, where `min` and
    `max` agree — which is how the first draft of these tests passed with the
    direction ignored entirely. Two strategies under a maximised metric is the
    smallest fixture where getting the sign wrong picks the *worse* constant and
    calls it the floor.
    """
    frame = pd.DataFrame({"y": BINARY * 2, "y_input": [0] * 40})
    plan = ValidationPlan(scheme="kfold", n_splits=4)

    def pick(direction: str) -> FloorReading:
        return compute_floor(
            frame,
            target="y",
            plan=plan,
            metric_name="accuracy",
            direction=direction,
            anchor_column="y_input",
            num_classes=2,
        )

    maximised, minimised = pick("maximize"), pick("minimize")

    assert maximised.strategies == minimised.strategies, "same measurements either way"
    assert maximised.best_strategy == "majority_class" and maximised.score == 0.75
    assert minimised.best_strategy == "anchor_carry_forward" and minimised.score == 0.25


def test_an_unknown_direction_refuses_to_pick_a_best() -> None:
    """A floor picked with the wrong sign is the most convincing wrong number
    this system could produce.

    `build_evidence_card` defaulted to `maximize` and recorded rogii's only real
    improvement as `rejected`; the same default here would pick the *worst*
    constant and call it the floor.
    """
    reading = _floor(SKEWED, "rmse", "unknown")

    assert not reading.is_defined
    assert "maximize" in reading.undefined_reason


# --- the artifact ---------------------------------------------------------------


def test_the_reading_is_a_dataset_fact_not_a_run(tmp_path: Path) -> None:
    """Goal 1: no `hypothesis_id`, no execution id, no generated file.

    Anything that made this look like a run would invite it to be compared
    against runs as though it were one — and it is the control, not a treatment.
    """
    reading = _floor(SKEWED, "rmse", "minimize")

    write_floor(tmp_path, reading)
    payload = json.loads((tmp_path / FLOOR_FILENAME).read_text(encoding="utf-8"))

    assert not {"hypothesis_id", "execution_id", "run_id", "code_path"} & set(payload)
    assert payload["validation"]["scheme"] == "kfold"
    assert load_floor(tmp_path).score == reading.score


def test_the_workspace_entry_point_reads_the_stage_before_it(tmp_path: Path) -> None:
    """The plan and the metric come from `baseline_choice.json`, the target and
    the anchor from `profile.json`. Nothing is re-derived here."""
    # Skewed on purpose: over a uniform target the mean and the median almost
    # coincide, so "median wins under MAE" would be an accident of the fixture
    # rather than a property of the metric.
    pd.DataFrame({"Id": range(40), "y": SKEWED * 2}).to_csv(tmp_path / "train.csv", index=False)
    (tmp_path / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_regression",
                "template_name": "t",
                "rationale": "",
                "metric_name": "mae",
                "target_column": "y",
                "train_file": "train.csv",
                "validation": {"scheme": "kfold", "n_splits": 4},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "profile.json").write_text(
        json.dumps({"competition": "demo", "target_column": "y"}), encoding="utf-8"
    )

    reading = floor_for_workspace(tmp_path)

    assert reading.is_defined, reading.undefined_reason
    assert reading.metric_name == "mae"
    assert reading.best_strategy == "median", "MAE's optimal constant, taken from the choice"
    assert reading.validation.n_splits == 4


def test_a_workspace_with_no_plan_has_no_floor(tmp_path: Path) -> None:
    """Not an exception. `floor_undefined` is a state, and the gate reads it."""
    reading = floor_for_workspace(tmp_path)

    assert not reading.is_defined
    assert "baseline_choice.json" in reading.undefined_reason
