from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"
    UNKNOWN = "unknown"


class MetricSpec(BaseModel):
    name: str
    direction: str = "maximize"  # maximize | minimize
    description: str = ""


class CompetitionSpec(BaseModel):
    slug: str
    title: str = ""
    description: str = ""
    evaluation_metric: MetricSpec | None = None
    problem_type: ProblemType = ProblemType.UNKNOWN
    submission_format: str = ""
    rules_url: str = ""
    data_url: str = ""
    deadline: str | None = None
    tags: list[str] = Field(default_factory=list)
    raw_html: str = ""
