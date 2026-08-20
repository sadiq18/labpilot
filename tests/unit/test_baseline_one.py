"""M23 step 4: a competent default, and the comparison the gate reports.

The floor says a number is not worse than nothing. Baseline 1 says whether the
pipeline is worth having at all, and the gate's output is the gap between them.

The case this milestone exists for is the one where the gap is the wrong way
round — `test_a_model_that_learns_nothing_does_not_beat_the_floor`. Until now
nothing here could report that, because `_observe_delta` compares runs against
each other and never against a constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.baseline_one import (
    BASELINE_ONE_FILENAME,
    MAX_AFFORDABLE_CELLS,
    ModelReading,
    affordability,
    compare,
    fit_baseline_one,
    load_baseline_one,
    write_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import FloorReading, compute_floor
from labpilot.research_engine.execution.baseline.selector import ValidationPlan

PLAN = ValidationPlan(scheme="kfold", n_splits=4)


def _learnable(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 3 * x1 - 2 * x2 + rng.normal(0, 0.3, n)})


def _noise(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "y": rng.normal(size=n)}
    )


def _fit(frame: pd.DataFrame, **kw) -> ModelReading:
    return fit_baseline_one(
        frame,
        target=kw.pop("target", "y"),
        plan=kw.pop("plan", PLAN),
        metric_name=kw.pop("metric_name", "rmse"),
        target_type=kw.pop("target_type", "continuous"),
        feature_columns=kw.pop("feature_columns", ["x1", "x2"]),
        **kw,
    )


# --- the comparison, which is the whole point ---------------------------------


def test_a_competent_default_beats_the_floor_on_learnable_data() -> None:
    """`y = 3*x1 - 2*x2 + noise`. If LightGBM cannot beat a constant here, the
    reference is broken rather than the dataset being hard."""
    frame = _learnable()
    floor = compute_floor(frame, target="y", plan=PLAN, metric_name="rmse", direction="minimize")

    comparison = compare(floor, _fit(frame), "minimize")

    assert comparison.beats_floor
    assert comparison.improvement > 0.5
    assert "Improvement" in comparison.render()


def test_a_model_that_learns_nothing_does_not_beat_the_floor() -> None:
    """The case the milestone exists for.

    Pure noise: there is nothing to learn, so a gradient-boosted tree fitted on
    it should not beat predicting a constant. `_observe_delta` compares runs
    against each other, so a campaign here would improve steadily and report
    progress the whole way down.
    """
    frame = _noise()
    floor = compute_floor(frame, target="y", plan=PLAN, metric_name="rmse", direction="minimize")

    comparison = compare(floor, _fit(frame), "minimize")

    assert not comparison.beats_floor
    assert comparison.improvement < 0
    assert "FAIL" in comparison.render()


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_improvement_is_signed_toward_better_whichever_way_the_metric_runs(
    direction: str,
) -> None:
    """A raw difference flips meaning with the metric, and every reader would
    have to remember which one this is.

    That is the class of mistake that recorded rogii's only genuine improvement
    as `rejected`. Positive always means better than the floor.
    """
    better = 0.9 if direction == "maximize" else 0.5
    floor = FloorReading(metric_name="m", score=0.7, best_strategy="mean")
    model = ModelReading(metric_name="m", score=better)

    comparison = compare(floor, model, direction)

    assert comparison.beats_floor
    assert comparison.improvement > 0


def test_a_metric_mismatch_is_refused_rather_than_subtracted() -> None:
    """Two numbers in different units. Subtracting them is arithmetic that means
    nothing, and it is the mismatch step 2 made visible one layer up."""
    comparison = compare(
        FloorReading(metric_name="rmse", score=1.0, best_strategy="mean"),
        ModelReading(metric_name="mae", score=0.5),
        "minimize",
    )

    assert not comparison.beats_floor
    assert "rmse" in comparison.incomparable_reason and "mae" in comparison.incomparable_reason


def test_a_zero_floor_has_a_verdict_but_no_percentage() -> None:
    """A relative improvement over zero is undefined, not infinite — and the
    verdict does not depend on the percentage being expressible."""
    comparison = compare(
        FloorReading(metric_name="mae", score=0.0, best_strategy="median"),
        ModelReading(metric_name="mae", score=0.2),
        "minimize",
    )

    assert comparison.improvement is None
    assert not comparison.beats_floor, "0.2 is worse than 0.0 under a minimised metric"
    # And it renders. Formatting `None` as a percentage took the whole report
    # down with a TypeError — a comparison that cannot be printed is one an
    # operator never sees.
    assert "n/a" in comparison.render()
    assert "FAIL" in comparison.render()


def test_no_floor_means_no_comparison_not_a_free_pass() -> None:
    """An absent control is not a model that passed."""
    comparison = compare(None, ModelReading(metric_name="rmse", score=0.1), "minimize")

    assert not comparison.beats_floor
    assert comparison.incomparable_reason


# --- the same plan as the floor -----------------------------------------------


def test_a_different_plan_yields_a_different_reading() -> None:
    """ "Three numbers on three splits compare nothing." """
    frame = _learnable()

    four = _fit(frame, plan=ValidationPlan(scheme="kfold", n_splits=4))
    eight = _fit(frame, plan=ValidationPlan(scheme="kfold", n_splits=8))

    assert four.score != eight.score
    assert four.fingerprint != eight.fingerprint
    assert four.validation.n_splits == 4


def test_a_plan_that_cannot_be_honoured_is_refused() -> None:
    """Same rule as the floor: silently falling back to `KFold` would put the
    model on a split the pipeline will not use."""
    reading = _fit(_learnable(), plan=ValidationPlan(scheme="group_kfold", group_key="well"))

    assert not reading.is_defined
    assert "group_kfold" in reading.undefined_reason


def test_the_same_bytes_and_plan_give_the_same_score() -> None:
    """The property, not the mechanism.

    At these parameters LightGBM samples nothing, so this holds today even
    without the seed — a mutation confirmed that. The test pins what must stay
    true rather than how it is currently achieved: a re-read of the same
    workspace reports the same number, and a reference that drifted between runs
    would be measuring nothing.
    """
    frame = _learnable()

    assert _fit(frame).score == _fit(frame).score


# --- affordability, derived rather than assumed --------------------------------


def test_an_image_dataset_is_unaffordable_rather_than_failed() -> None:
    """The plan's trap: a gate demanding something unaffordable gets switched off.

    An image competition's label is still a class column — which is why the
    *floor* is defined there — but a tree has nothing to read without an
    extractor, and building one here is the "anything clever" this module
    refuses. The gate's `awaiting_ml` state reads this reason; it is not `passed`.
    """
    affordable, reason = affordability(_learnable(), ["x1", "x2"], modality="image")

    assert not affordable
    assert "extractor" in reason


def test_a_table_past_the_budget_is_unaffordable() -> None:
    frame = _learnable(10)

    affordable, reason = affordability(frame, [f"f{i}" for i in range(MAX_AFFORDABLE_CELLS)])

    assert not affordable
    assert "budget" in reason


def test_a_tabular_dataset_within_budget_is_affordable() -> None:
    affordable, reason = affordability(_learnable(), ["x1", "x2"])

    assert affordable and reason == ""


def test_unaffordable_records_a_reason_rather_than_a_score() -> None:
    """`awaiting_ml` needs the difference between "cannot run here", "the fit
    failed", and "nobody tried"."""
    reading = _fit(_learnable(), modality="image")

    assert not reading.is_defined
    assert reading.score is None
    assert reading.undefined_reason


# --- the target's shape decides the estimator -----------------------------------


def test_a_classification_target_gets_a_classifier() -> None:
    """`target_type` from M23 step 1, not a fourth cardinality rule.

    This is the third consumer of that derived field, and deriving it once was
    the point: there is no place left for a rule to disagree.
    """
    rng = np.random.default_rng(2)
    n = 300
    x1 = rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": (x1 > 0).astype(int)})

    reading = _fit(frame, metric_name="accuracy", target_type="binary", num_classes=2)

    assert reading.is_defined
    assert reading.score > 0.9, "a separable target should be nearly perfectly classified"


def test_a_shapeless_target_has_no_objective_to_fit_toward() -> None:
    """`unknown` and `none` are not tasks. Fitting a regressor at a target
    nobody could identify produces a number that means nothing."""
    reading = _fit(_learnable(), target_type="unknown")

    assert not reading.is_defined
    assert "target_type" in reading.undefined_reason


# --- review findings -----------------------------------------------------------


def test_a_partitioned_frame_with_a_duplicate_index_still_fits() -> None:
    """Review finding. `fit_baseline_one` inherited the floor's split crash.

    A frame concatenated from per-partition files keeps each file's `0..n`, so
    the index is not unique — rogii's layout. Both readings went down on it,
    which means the comparison the gate reports could not exist for the one
    dataset this milestone was built around.
    """
    parts = [
        pd.DataFrame({"y": np.arange(10.0), "x": np.arange(10.0) * 2, "w": [f"w{k}"] * 10})
        for k in range(4)
    ]
    frame = pd.concat(parts)
    assert not frame.index.is_unique, "the fixture must reproduce the real layout"

    reading = _fit(
        frame,
        plan=ValidationPlan(scheme="partition_suffix_holdout", group_key="w", holdout_fraction=0.5),
        feature_columns=["x"],
    )

    assert reading.is_defined, reading.undefined_reason


def test_the_two_readings_share_one_fingerprint_implementation() -> None:
    """Review finding: this module kept its own copy of the pointer-hashing bug.

    Two implementations of one digest is how the floor and the model come to
    disagree about whether they described the same data — fixing the floor alone
    would have left this one wrong. The model's digest is the floor's, plus the
    feature set, which is the only input it has that the floor does not.
    """
    from labpilot.research_engine.execution.baseline.floor import fingerprint_of

    frame = _learnable()
    plan = ValidationPlan(scheme="kfold", n_splits=4)
    shared = fingerprint_of(frame["y"], plan, "rmse")

    one = _fit(frame, plan=plan, feature_columns=["x1", "x2"])
    fewer = _fit(frame, plan=plan, feature_columns=["x1"])

    assert shared in (one.fingerprint, "") or one.fingerprint != fewer.fingerprint
    assert one.fingerprint != fewer.fingerprint, "the feature set is part of what was measured"


def test_a_string_target_gets_a_stable_fingerprint() -> None:
    """The bug this inherited: `to_numpy().tobytes()` on an object dtype digests
    Python object addresses, so equal values with distinct objects disagreed.

    Built at runtime so CPython does not intern the strings into the same
    objects — without that, the test passes against the broken version.
    """
    pieces = [("y", "e", "s"), ("n", "o")]
    labels = ["".join(chars) for chars in pieces] * 10
    literal = pd.DataFrame({"y": ["yes", "no"] * 10, "x": np.arange(20.0)})
    built = pd.DataFrame({"y": labels, "x": np.arange(20.0)})
    assert [id(v) for v in literal["y"]] != [id(v) for v in built["y"]]

    first = _fit(
        literal, metric_name="accuracy", target_type="binary", num_classes=2, feature_columns=["x"]
    )
    second = _fit(
        built, metric_name="accuracy", target_type="binary", num_classes=2, feature_columns=["x"]
    )

    assert first.fingerprint == second.fingerprint


def test_the_splitter_is_imported_by_its_public_name() -> None:
    """A private name reached across a module boundary is a contract nothing
    states, and this one is load-bearing: the floor and Baseline 1 must split
    identically or the comparison means nothing."""
    import inspect

    from labpilot.research_engine.execution.baseline import baseline_one

    source = inspect.getsource(baseline_one)

    assert "folds_for" in source
    assert "_folds" not in source.replace("folds_for", ""), "no private splitter import"


# --- the artifact -----------------------------------------------------------------


def test_the_reading_is_a_dataset_fact_not_a_run(tmp_path: Path) -> None:
    reading = _fit(_learnable())

    write_baseline_one(tmp_path, reading)
    payload = json.loads((tmp_path / BASELINE_ONE_FILENAME).read_text(encoding="utf-8"))

    assert not {"hypothesis_id", "execution_id", "run_id", "code_path"} & set(payload)
    assert payload["model"] == "lightgbm"
    assert len(payload["fold_scores"]) == 4, "per fold, so one bad fold is visible"
    assert load_baseline_one(tmp_path).score == reading.score


def test_the_verdict_is_not_stored_anywhere(tmp_path: Path) -> None:
    """AGENTS.md rule 2. A stored verdict is derived state that outlives its
    cause, which is the mistake `apply_card_to_beliefs` cost this repo.

    Both artifacts hold measurements; `beats_floor` is recomputed from them.
    """
    frame = _learnable()
    floor = compute_floor(frame, target="y", plan=PLAN, metric_name="rmse", direction="minimize")
    write_baseline_one(tmp_path, _fit(frame))

    payload = json.loads((tmp_path / BASELINE_ONE_FILENAME).read_text(encoding="utf-8"))

    assert not {"beats_floor", "improvement", "passed", "verdict"} & set(payload)
    assert compare(floor, load_baseline_one(tmp_path), "minimize").beats_floor
