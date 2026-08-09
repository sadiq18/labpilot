"""Normalized literature catalog + research-knowledge models (design §4.5–4.6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from labpilot.research_engine.intelligence.feature_recipes import FeatureRecipe


class Paper(BaseModel):
    """Normalized catalog entry — every backend maps into this shape."""

    id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citations: int | None = None
    concepts: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    pdf_path: str | None = None
    github_urls: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    arxiv_id: str | None = None
    doi: str | None = None
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    urls: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)

    def rank_score(self, *, alpha: float | None = None, as_of_year: int | None = None) -> float:
        """relevance × citation-velocity factor × age decay.

        See :mod:`labpilot.research_engine.intelligence.literature.ranking`.
        """
        from labpilot.research_engine.intelligence.literature.ranking import (
            DEFAULT_AGE_ALPHA,
            rank_score,
        )

        return rank_score(
            self,
            alpha=DEFAULT_AGE_ALPHA if alpha is None else alpha,
            as_of_year=as_of_year,
        )

    def bucket(self, *, recent_years: int | None = None, as_of_year: int | None = None) -> str:
        from labpilot.research_engine.intelligence.literature.ranking import (
            DEFAULT_RECENT_YEARS,
            is_recent,
        )

        recent_years = DEFAULT_RECENT_YEARS if recent_years is None else recent_years

        return (
            "recent"
            if is_recent(self, recent_years=recent_years, as_of_year=as_of_year)
            else "foundational"
        )


class PaperKnowledge(BaseModel):
    """Extracted research knowledge — NOT a paper summary (design §4.6)."""

    paper_id: str = ""
    title: str = ""
    contributions: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    ideas_worth_testing: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    feature_recipes: list[FeatureRecipe] = Field(default_factory=list)
    datasets_used: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    grounded_in: Literal["abstract", "pdf_excerpt", "metadata"] = "abstract"

    @field_validator("grounded_in", mode="before")
    @classmethod
    def _blank_means_omitted(cls, value: Any) -> Any:
        """An empty string is the model declining to answer, not a new value.

        Measured on rogii 2026-08-09: `PaperAnalyzerAgent` returned
        `"grounded_in": ""`, failed validation three times, and the paper's
        knowledge was dropped entirely — one blank enum against thirteen
        populated fields. Omitting the key was always accepted; writing it
        empty is the same statement, so it gets the same answer.

        Only blank coerces. An unrecognized *value* — "full_text", say — is a
        genuine disagreement about provenance, and silently recording it as
        "abstract" would misstate what the extraction was grounded in. That
        still fails, loudly.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return "abstract"
        return value
