"""The five questions, answered by scoring rather than by a chain of `if`s.

M22 step 3, and the first step allowed to change an answer. What it must not
change is an answer the profiler already gets right: the shapes that work today
resolve to the same target and the same key, and now say how sure they are.

What is new is the other three answers — which columns are usable features,
how the scored units relate to the training ones, and what the dataset is
scored by — and a dataset with none of Kaggle's inputs finally producing a
schema instead of an exception.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from helpers.dataset_shapes import (
    build_bool_target,
    build_no_kaggle_inputs,
    build_partition_suffix,
    build_partitioned_with_template,
    build_strong_signals,
    build_template_only,
)
from helpers.dataset_sources import DictSource

from labpilot.accessor.profiler.schema import MetricRef
from labpilot.accessor.profiler.source import DeclaredFacts
from labpilot.accessor.profiler.tabular import DatasetProfile, TabularProfiler
from labpilot.config import ProfilerConfig


def _profile(data_dir: Path) -> DatasetProfile:
    return TabularProfiler(ProfilerConfig()).profile_directory(data_dir, data_dir.name)


# --- goal 4: no regression where the profiler is already right --------------


@pytest.mark.parametrize(
    ("shape", "target", "key"),
    [
        ("strong_signals", "SalePrice", "Id"),
        ("bool_target", "Transported", "PassengerId"),
        ("titanic", "Survived", "PassengerId"),
    ],
)
def test_the_shapes_that_work_keep_their_answers(
    request, tmp_path: Path, shape: str, target: str, key: str
) -> None:
    """house-prices, spaceship-titanic and titanic: same values, now with evidence.

    The scoring rewrite has to be invisible where the old procedure was right,
    or it is not a rewrite of *how* the answer is reached — it is a change of
    answer wearing that description.
    """
    builders = {"strong_signals": build_strong_signals, "bool_target": build_bool_target}
    data_dir = (
        builders[shape](tmp_path)
        if shape in builders
        else request.getfixturevalue("titanic_data_dir")
    )
    profile = _profile(data_dir)

    assert profile.target_column == target
    assert profile.id_columns == [key]
    assert profile.id_column == key, "the singular field is a view over the list"
    assert profile.confidence_in("target_column") >= 0.85
    assert profile.confidence_in("id_columns") >= 0.85


# --- features, and the exclusion that matters -------------------------------


def test_a_column_identical_to_the_target_is_not_a_feature(tmp_path: Path) -> None:
    """The leak that looks like the best feature in the dataset.

    rogii's `TVT_input` equals the target wherever both are present and is
    absent exactly on the scored rows. As a plain feature a model learns to copy
    it and then meets NaN on every row it has to predict — and the CV score
    *improves*, which is why nothing downstream catches this.
    """
    profile = _profile(build_partitioned_with_template(tmp_path))

    assert profile.anchor_column == "Depth_input"
    assert profile.excluded_columns["Depth_input"] == "equals_target"
    assert "Depth_input" not in profile.feature_columns
    assert profile.excluded_columns["Depth"] == "is_target"
    assert profile.excluded_columns["Zone_Depth"] == "unavailable_at_scoring"
    assert profile.excluded_columns["id"] == "is_id"
    assert profile.feature_columns == ["md", "azimuth", "formation"]


def test_features_are_derived_from_the_exclusions(tmp_path: Path) -> None:
    """One fact, one place. A stored feature list that drifted from the
    exclusions would be a leak with no symptom."""
    profile = _profile(build_strong_signals(tmp_path))

    assert set(profile.feature_columns) == {column.name for column in profile.columns} - set(
        profile.excluded_columns
    )
    assert "SalePrice" not in profile.feature_columns
    assert "Id" not in profile.feature_columns


# --- how the scored units relate to the training ones -----------------------


def test_the_split_relationship_is_concluded_with_evidence(tmp_path: Path) -> None:
    """Three shapes, three conclusions, each carrying what it was concluded from."""
    flat = _profile(build_strong_signals(tmp_path))
    none_at_all = _profile(build_no_kaggle_inputs(tmp_path))

    assert flat.train_test_relationship == "disjoint_units"
    assert none_at_all.train_test_relationship == "no_test_provided"
    # "IID" is the residual — what is left when nothing else fired — so it is
    # actionable and never assertable. A missing scoring input is a fact.
    assert flat.confidence_in("train_test_relationship") <= 0.75
    assert none_at_all.confidence_in("train_test_relationship") >= 0.85


def test_a_forecast_split_is_recognised_as_one(tmp_path: Path) -> None:
    """A partition-tail split is the one that makes a random CV meaningless."""
    profile = _profile(build_partition_suffix(tmp_path))

    assert profile.scored_is_partition_suffix
    assert profile.train_test_relationship == "partition_suffix"
    assert profile.confidence_in("train_test_relationship") >= 0.60


# --- the objective ----------------------------------------------------------


def test_a_declared_metric_is_recorded_with_its_source() -> None:
    """The profiler records how the metric was reached; it does not resolve one.

    Canonicalisation lives in the metric registry, which is in `research_engine`
    — a layer `accessor` may not import. A source states what it was told, and
    the evidence says the statement is where the answer came from.
    """
    frame = pd.DataFrame({"Id": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
    source = DictSource(
        {
            "train.csv": frame,
            "test.csv": frame[["Id", "x"]],
            "sample_submission.csv": frame[["Id", "y"]],
        },
        DeclaredFacts(metric=MetricRef(name="RMSE", key="rmse", direction="minimize")),
    )

    profile = TabularProfiler(ProfilerConfig()).profile_dataset(source, "declared-metric")

    assert profile.metric is not None
    assert (profile.metric.key, profile.metric.direction) == ("rmse", "minimize")
    assert profile.metric.source == "declared"
    # Declared *and* pointed in a direction: half a metric is worth less.
    assert profile.confidence_in("metric") >= 0.85


def test_position_alone_never_answers(tmp_path: Path) -> None:
    """Nothing withheld, so only the template's column order is left.

    This is the branch five rounds of PR #117 kept coming back to: `overlap[1]`
    works on aerial-cactus by convention and would pick `id` from a reversed
    header. It is not improved here, it is **capped** — the value is still
    recorded, and it lands `uncertain` so step 4 asks instead of acting.
    """
    profile = _profile(build_template_only(tmp_path))
    target = profile.inferences["target_column"]

    assert profile.target_column == "y"
    assert [signal.id for signal in target.signals][0] == "positional_template_overlap"
    assert target.band == "uncertain"
    assert target.confidence <= 0.50


def test_an_undeclared_metric_is_not_invented(tmp_path: Path) -> None:
    """No metric is the honest answer when nothing states one."""
    profile = _profile(build_strong_signals(tmp_path))

    assert profile.metric is None
    assert profile.confidence_in("metric") == 0.0
    assert profile.inferences["metric"].band == "uncertain"
