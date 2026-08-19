from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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
    """A competition's evaluation metric: what it is, and which way is better.

    `direction` defaulted to `"maximize"`, so every unstated direction claimed
    the one that inverts every verdict for a loss. rogii's fifteen evidence
    cards were built that way — the single genuine improvement recorded as
    `rejected`. The default is now `"unknown"`, which is what a contract that
    never said actually means, and which callers can see.
    """

    name: str
    direction: Literal["maximize", "minimize", "unknown"] = "unknown"
    description: str = ""
    # Canonical key used by BaselineSelector and training templates (e.g.
    # "accuracy", "auc", "rmse"). None when the raw metric string could not be
    # mapped to a catalogued evaluator key.
    key: str | None = None

    @field_validator("direction", mode="before")
    @classmethod
    def _accept_any_spelling(cls, raw: object) -> object:
        """`min`, `Minimize`, `maximise` — every form `direction.py` reads.

        Narrowing this field to a `Literal` without normalising first made the
        model stricter than the reader that has always accepted those spellings,
        so a hand-written competition config raised `ValidationError` out of
        `CompetitionParser` — and inside `capability.py`'s broad `except
        Exception`, silently discarded the entire analyze-derived contract.

        `None` and missing stay untouched so the field default applies.
        """
        if raw is None:
            return raw
        from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
            normalize_direction,
        )

        return normalize_direction(raw)

    @model_validator(mode="after")
    def _fill_direction_from_key(self) -> "MetricSpec":
        """A catalogued key already knows which way is better.

        Fills only an *unknown* direction. A stated direction that contradicts
        the registry is deliberately left standing rather than corrected: it is
        the contradiction `resolve_objective` blocks the campaign on, and
        silently rewriting it here would delete the detection while leaving the
        wrong contract on disk.
        """
        if self.direction != "unknown" or self.key is None:
            return self
        from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
            direction_of,
        )

        measured = direction_of(self.key)
        if measured is not None:
            object.__setattr__(self, "direction", measured)
        return self


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
