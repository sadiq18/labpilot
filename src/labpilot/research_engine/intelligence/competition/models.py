from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from labpilot.accessor.kaggle.models import CompetitionMetadata

__all__ = [
    "CompetitionMetadata",
    "CompetitionSpec",
    "MetricSpec",
    "ProblemType",
]


class ProblemType(StrEnum):
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    UNKNOWN = "unknown"


class MetricSpec(BaseModel):
    name: str
    direction: str = "maximize"  # maximize | minimize
    description: str = ""
    # Canonical key used by BaselineSelector and training templates (e.g.
    # "accuracy", "auc", "rmse"). None when the raw Kaggle metric string
    # could not be mapped to a supported evaluator key.
    key: str | None = None


class CompetitionSpec(BaseModel):
    slug: str
    title: str = ""
    description: str = ""
    evaluation_metric: MetricSpec | None = None
    problem_type: ProblemType = ProblemType.UNKNOWN
    submission_format: str = ""
    submission_columns: list[str] = Field(default_factory=list)
    rules_url: str = ""
    data_url: str = ""
    deadline: str | None = None
    max_daily_submissions: int | None = None
    submissions_disabled: bool = False
    is_kernels_submissions_only: bool = False
    submission_mode: Literal["csv", "kernel"] = "csv"
    kernel_output_file: str = "submission.csv"
    submissions_url: str = ""
    tags: list[str] = Field(default_factory=list)
    raw_html: str = ""
    baseline_strategy: str = "lightweight"  # lightweight | deep (P1 opt-in)

    # Optional overrides for competitions whose file names don't follow the
    # "train*/test*/*submission*" convention the profiler assumes by default.
    train_file_pattern: str = "train"
    test_file_pattern: str = "test"
    submission_file_pattern: str = "submission"
