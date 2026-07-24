"""Normalized literature catalog + research-knowledge models (design §4.5–4.6)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    datasets_used: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    grounded_in: Literal["abstract", "pdf_excerpt", "metadata"] = "abstract"
