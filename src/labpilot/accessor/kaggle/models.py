"""Raw Kaggle API DTOs shared by accessor clients."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CompetitionMetadata(BaseModel):
    """Raw metadata resolved from the Kaggle API for a competition slug.

    Smaller than :class:`~labpilot.research_engine.intelligence.competition.models.CompetitionSpec`
    — what fetchers return before parser enrichment.
    """

    slug: str
    title: str = ""
    description: str = ""
    category: str = ""
    evaluation_metric_raw: str = ""
    deadline: str | None = None
    max_daily_submissions: int | None = None
    submissions_disabled: bool = False
    is_kernels_submissions_only: bool = False
    tags: list[str] = Field(default_factory=list)
