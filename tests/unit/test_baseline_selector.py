"""Tests for metadata problem-type inference + baseline selection."""

from __future__ import annotations

import pytest

from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile
from labpilot.research_engine.execution.baseline.selector import BaselineSelector
from labpilot.research_engine.intelligence.competition.infer_problem_type import (
    infer_problem_type_from_metadata,
)
from labpilot.research_engine.intelligence.competition.models import (
    CompetitionSpec,
    MetricSpec,
    ProblemType,
)


def test_infer_biohub_tracking_as_image() -> None:
    inferred = infer_problem_type_from_metadata(
        title="Biohub - Cell Tracking During Development",
        description="Detect and track zebrafish cells through 3D space and time",
        tags=["Research", "biology", "computer vision", "image", "video", "object detection"],
    )
    assert inferred is ProblemType.IMAGE_CLASSIFICATION


def test_infer_titanic_style_classification() -> None:
    inferred = infer_problem_type_from_metadata(
        title="Titanic",
        description="Binary classification survival prediction",
        tags=["tabular", "classification"],
    )
    assert inferred is ProblemType.TABULAR_CLASSIFICATION


def test_infer_rogii_mse_as_tabular_regression() -> None:
    inferred = infer_problem_type_from_metadata(
        title="ROGII - Wellbore Geology Prediction",
        description="Predict wellbore geology (tvt) from well log data",
        tags=["Featured", "geology", "multimodal", "evaluation", "mean squared error"],
        metric_name="Mean Squared Error",
        metric_description="MSE",
    )
    assert inferred is ProblemType.TABULAR_REGRESSION


def test_selector_prefers_mse_metadata_over_image_modality() -> None:
    competition = CompetitionSpec(
        slug="rogii-wellbore-geology-prediction",
        title="ROGII - Wellbore Geology Prediction",
        tags=["geology", "multimodal", "mean squared error"],
        evaluation_metric=MetricSpec(name="Mean Squared Error", direction="minimize", key="mse"),
    )
    profile = DatasetProfile(
        competition=competition.slug,
        modality="image",
        files=["train/a.png", "train/a__typewell.csv"],
        warnings=["images_in=train"],
    )
    choice = BaselineSelector().select(competition, profile)
    assert choice.problem_type == ProblemType.TABULAR_REGRESSION.value
    assert choice.metric_name == "mse"


def test_selector_uses_metadata_when_profile_empty() -> None:
    competition = CompetitionSpec(
        slug="biohub-cell-tracking-during-development",
        title="Biohub - Cell Tracking During Development",
        description="Detect and track zebrafish cells through 3D space and time",
        tags=["computer vision", "image", "video", "object detection"],
        evaluation_metric=MetricSpec(name="czi_biohub_zebrafish_133605", direction="maximize"),
    )
    profile = DatasetProfile(competition=competition.slug)  # empty
    choice = BaselineSelector().select(competition, profile)
    assert choice.problem_type == ProblemType.IMAGE_CLASSIFICATION.value
    assert choice.template_name == "image_classification"


def test_selector_refuses_empty_unknown() -> None:
    competition = CompetitionSpec(slug="mystery-comp")
    profile = DatasetProfile(competition="mystery-comp")
    with pytest.raises(ValueError, match="Cannot infer problem type"):
        BaselineSelector().select(competition, profile)


def test_selector_uses_competition_metric_key_when_supported() -> None:
    competition = CompetitionSpec(
        slug="test",
        evaluation_metric=MetricSpec(
            name="area_under_the_roc_curve",
            direction="maximize",
            key="auc",
        ),
    )
    profile = DatasetProfile(
        competition="test",
        files=["train.csv"],
        row_count=100,
        column_count=3,
        columns=[
            ColumnProfile(name="id", dtype="int64", null_pct=0, unique_count=100, is_numeric=True),
            ColumnProfile(
                name="target", dtype="int64", null_pct=0, unique_count=2, is_numeric=True
            ),
        ],
        target_column="target",
        id_column="id",
    )
    choice = BaselineSelector().select(competition, profile)
    assert choice.metric_name == "auc"


def test_the_workspace_contract_does_not_claim_maximize_for_a_loss(tmp_path) -> None:
    """Review finding, at the one call site that restated the removed default.

    `_ensure_competition_json` hydrated the workspace contract with
    `direction=str(metric_raw.get("direction") or "maximize")`. Because
    `MetricSpec` derives a direction only when the stated one is unknown, that
    literal blocked the derivation — so an analyze report omitting `direction`
    wrote `maximize` for RMSE into the file every later stage reads. That is the
    rogii inversion surviving the change meant to remove it.
    """
    import inspect

    from labpilot.research_engine.execution.capabilities.workspace import capability
    from labpilot.research_engine.intelligence.competition.models import MetricSpec

    source = inspect.getsource(capability)
    assert 'or "maximize"' not in source, "the removed default is restated here"

    metric_raw = {"name": "Root Mean Squared Error", "key": "rmse"}
    metric = MetricSpec(
        name=str(metric_raw.get("name")),
        direction=metric_raw.get("direction") or "unknown",
        description="",
        key=metric_raw.get("key"),
    )

    assert metric.direction == "minimize"
