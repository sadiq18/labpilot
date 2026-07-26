from labpilot.research_engine.execution.baseline.selector import BaselineSelector
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec, MetricSpec, ProblemType
from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile


def _profile() -> DatasetProfile:
    return DatasetProfile(
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


def test_selector_uses_competition_metric_key_when_supported():
    competition = CompetitionSpec(
        slug="test",
        evaluation_metric=MetricSpec(
            name="area_under_the_roc_curve",
            direction="maximize",
            key="auc",
        ),
    )
    choice = BaselineSelector().select(competition, _profile())
    assert choice.metric_name == "auc"


def test_selector_falls_back_for_unsupported_key():
    competition = CompetitionSpec(
        slug="test",
        evaluation_metric=MetricSpec(
            name="quadratic_weighted_kappa",
            direction="maximize",
            key=None,
        ),
    )
    choice = BaselineSelector().select(competition, _profile())
    assert choice.metric_name == "accuracy"
