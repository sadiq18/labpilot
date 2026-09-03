"""M23 step 7: the floor is the control, and something finally produces it.

Two things land here. The readings get a caller — until now `compute_floor` and
`fit_baseline_one` were reachable only from tests, so no campaign ever wrote
`baseline_floor.json` and the gate would have reported `floor_missing` forever.
And the baseline plan's COMPARE gets a control, which is goal 2.

The restraint is the design's, and it is the point: **no third reading on
`ObservedOutcomes`**. The floor arrives as `parent_cv`, so `_decide` — the single
funnel for every verdict in this system — is untouched, and a metric mismatch is
caught for free by machinery that already exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from labpilot.research_engine.execution.baseline.baseline_one import load_baseline_one
from labpilot.research_engine.execution.baseline.floor import load_floor
from labpilot.research_engine.execution.baseline.gate import evaluate_gate, reading_fingerprint
from labpilot.research_engine.execution.baseline.runner import ensure_readings, floor_as_control


def _workspace(tmp_path: Path, *, learnable: bool = True) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    y = 3 * x1 + rng.normal(0, 0.3, n) if learnable else rng.normal(size=n)
    pd.DataFrame({"Id": range(n), "x1": x1, "x2": rng.normal(size=n), "y": y}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_regression",
                "template_name": "t",
                "rationale": "",
                "metric_name": "rmse",
                "target_column": "y",
                "train_file": "train.csv",
                "validation": {"scheme": "kfold", "n_splits": 4},
            }
        ),
        encoding="utf-8",
    )
    # A *serialized* `DatasetProfile`, not a hand-built dict. `target_type`,
    # `modality` and `feature_columns` are computed fields, so a hand-written
    # profile silently lacks them — and every test here would then be asserting
    # against an artifact this system never produces.
    from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile

    profile = DatasetProfile(
        competition="demo",
        schema_version=4,
        target_column="y",
        id_columns=["Id"],
        row_count=n,
        train_file="train.csv",
        modalities=[{"modality": "tabular", "present": True, "role": "primary"}],
        columns=[
            ColumnProfile(name="Id", dtype="int64", unique_count=n, is_numeric=True),
            ColumnProfile(name="x1", dtype="float64", unique_count=n, is_numeric=True),
            ColumnProfile(name="x2", dtype="float64", unique_count=n, is_numeric=True),
            ColumnProfile(
                name="y",
                dtype="float64",
                unique_count=n,
                is_numeric=True,
                stats={"min": -9.0, "max": 9.0},
            ),
        ],
    )
    (tmp_path / "profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    return tmp_path


# --- something produces the readings at last ----------------------------------


def test_the_readings_get_written_by_something_other_than_a_test(tmp_path: Path) -> None:
    """`compute_floor` and `fit_baseline_one` had no caller in `src/`.

    Both artifacts existed only in unit tests, so a real campaign would have
    reached the gate and been told `floor_missing` forever. This is the caller
    that was missing.
    """
    _workspace(tmp_path)

    floor, model = ensure_readings(tmp_path)

    assert floor is not None and floor.is_defined
    assert model is not None and model.is_defined
    assert (tmp_path / "baseline_floor.json").is_file()
    assert (tmp_path / "baseline_one.json").is_file()


def test_the_readings_are_stamped_so_they_can_go_stale(tmp_path: Path) -> None:
    """A reading nobody stamped cannot be invalidated by an answer, and the gate
    would keep reporting over a measurement of a rejected column."""
    _workspace(tmp_path)

    ensure_readings(tmp_path)

    stamp = reading_fingerprint(tmp_path)
    assert load_floor(tmp_path).workspace_fingerprint == stamp
    assert load_baseline_one(tmp_path).workspace_fingerprint == stamp


def test_a_current_reading_is_not_recomputed(tmp_path: Path) -> None:
    """Five LightGBM fits is real time, and a control that re-measured itself on
    every read would give the same number while giving nobody a way to notice."""
    _workspace(tmp_path)
    first, _ = ensure_readings(tmp_path)
    written = (tmp_path / "baseline_floor.json").read_text(encoding="utf-8")

    second, _ = ensure_readings(tmp_path)

    assert second.computed_at == first.computed_at, "not re-measured"
    assert (tmp_path / "baseline_floor.json").read_text(encoding="utf-8") == written


def test_an_answer_makes_the_readings_be_taken_again(tmp_path: Path) -> None:
    """The other half: reuse must not outlive the workspace it described."""
    from labpilot.accessor.profiler.questions import ANSWERS_FILENAME

    _workspace(tmp_path)
    first, _ = ensure_readings(tmp_path)

    (tmp_path / ANSWERS_FILENAME).write_text(json.dumps({"target_column": "y"}), encoding="utf-8")
    second, _ = ensure_readings(tmp_path)

    assert second.workspace_fingerprint != first.workspace_fingerprint


def test_a_workspace_with_no_plan_yields_no_readings(tmp_path: Path) -> None:
    """Not an exception. The gate has nine states and none of them is
    "the baseline crashed"."""
    floor, model = ensure_readings(tmp_path)

    assert floor is None or not floor.is_defined
    assert model is None or not model.is_defined


def test_a_profile_with_one_malformed_corner_still_fits(tmp_path: Path) -> None:
    """Validation is all-or-nothing, and the gate reads the result as a fact.

    A legacy `modalities` entry missing `role` is enough to make
    `DatasetProfile.model_validate` raise. `_fit` then returned None, which the
    gate reports as `awaiting_ml` — and `awaiting_ml` is now one of the states a
    campaign may move past, so an unparseable profile would have read as
    "Baseline 1 is merely unaffordable here" and waved the campaign through.

    Found by writing a fixture with exactly that defect while testing something
    else, which is the only reason it is covered at all.
    """
    _workspace(tmp_path)
    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    profile["modalities"] = [{"modality": "tabular", "present": True, "confidence": 0.9}]
    assert "target_type" in profile, "the stored fields are what the fallback reads"
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    _floor, model = ensure_readings(tmp_path)

    assert model is not None and model.is_defined, model.undefined_reason if model else "no model"


def test_a_profile_that_predates_the_computed_fields_still_fits(tmp_path: Path) -> None:
    """The other direction: validation is what *derives* those fields.

    A profile written before `target_type` existed does not carry it, so reading
    the raw dict alone would give `unknown` and refuse to fit. The model is tried
    first for exactly this reason.
    """
    _workspace(tmp_path)
    profile = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    for computed in ("target_type", "modality", "feature_columns"):
        profile.pop(computed, None)
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    _floor, model = ensure_readings(tmp_path)

    assert model is not None and model.is_defined, model.undefined_reason if model else "no model"


# --- the floor, shaped as a control -------------------------------------------


def test_the_floor_arrives_as_a_cv_metric(tmp_path: Path) -> None:
    """`{"cv_rmse": 3.4}` — the shape `_primary_cv_keyed` reads.

    Shaped this way on purpose: `_decide` is the single funnel for every verdict,
    so a control that looks like every other control means the gain, the sign,
    the card and the hypothesis status all work unchanged.
    """
    _workspace(tmp_path)

    floor, _model = ensure_readings(tmp_path)
    metrics = floor_as_control(floor)

    assert list(metrics) == ["cv_rmse"]
    assert metrics["cv_rmse"] > 0


def test_the_floor_control_carries_the_metric_in_its_key(tmp_path: Path) -> None:
    """The key is not bookkeeping. Two runs yielding numbers from different keys
    is how six rogii cards recorded a "gain" of -194.30 by subtracting an
    accuracy from an RMSE, and `_same_metric` refuses that — but only if the key
    names the metric.
    """
    _workspace(tmp_path)
    choice = json.loads((tmp_path / "baseline_choice.json").read_text(encoding="utf-8"))
    choice["metric_name"] = "mae"
    (tmp_path / "baseline_choice.json").write_text(json.dumps(choice), encoding="utf-8")

    floor, _model = ensure_readings(tmp_path)
    metrics = floor_as_control(floor)

    assert list(metrics) == ["cv_mae"]


def test_no_floor_means_no_control_rather_than_a_zero(tmp_path: Path) -> None:
    """A control of 0.0 is one every model beats, which would turn the gate into
    a rubber stamp — the opposite of what it is for."""
    floor, _model = ensure_readings(tmp_path)

    assert floor_as_control(floor) == {}
    assert floor_as_control(None) == {}, "and an absent reading is not a zero either"


# --- the verdict the gate reaches once both exist --------------------------------


def test_the_gate_reaches_a_real_verdict_on_a_produced_workspace(tmp_path: Path) -> None:
    """End to end from artifacts alone: readings produced, then judged."""
    _workspace(tmp_path, learnable=True)
    ensure_readings(tmp_path)

    assert evaluate_gate(tmp_path).state == "passed"


def test_a_pipeline_that_cannot_beat_a_constant_reaches_failed(tmp_path: Path) -> None:
    """Pure noise: there is nothing to learn, so the generic model should not
    beat predicting a constant, and the gate should say so."""
    _workspace(tmp_path, learnable=False)
    ensure_readings(tmp_path)

    verdict = evaluate_gate(tmp_path)

    assert verdict.state == "failed"
    assert verdict.blocks_research
    assert not verdict.withholds_anything, "still observe-only until step 8"
