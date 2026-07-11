from enum import StrEnum

from pydantic import BaseModel, Field


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


class CompetitionMetadata(BaseModel):
    """Raw metadata resolved from the Kaggle API for a competition slug.

    This is intentionally smaller than `CompetitionSpec` — it's what
    `CompetitionMetadataFetcher` implementations return, before it gets
    turned into a `CompetitionSpec` by `CompetitionParser`.
    """

    slug: str
    title: str = ""
    description: str = ""
    category: str = ""
    evaluation_metric_raw: str = ""


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
    tags: list[str] = Field(default_factory=list)
    raw_html: str = ""

    # Optional overrides for competitions whose file names don't follow the
    # "train*/test*/*submission*" convention the profiler assumes by default.
    # The Kaggle API's competition search doesn't expose file names, so this
    # still requires a local override (see configs/competitions/README.md).
    train_file_pattern: str = "train"
    test_file_pattern: str = "test"
    submission_file_pattern: str = "submission"
