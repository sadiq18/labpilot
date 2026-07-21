"""Shared data models for the Research Intelligence Platform.

``ResearchArtifact`` is the most important abstraction (design §3.1): every
paper, experiment, blog, repository, forum thread, winning solution, or note
becomes one — a single interface so downstream code (KB upsert, retrieval,
Hypothesis Assistant) never special-cases providers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class ResearchArtifactType(StrEnum):
    """Every source kind maps to one of these."""

    PAPER = "paper"
    EXPERIMENT = "experiment"
    BLOG = "blog"
    REPOSITORY = "repository"
    DISCUSSION = "discussion"  # forum thread / GitHub issue / Reddit
    NOTE = "note"  # manual / imported note
    COMPETITION = "competition"  # related-comp or profile slice
    WINNING_SOLUTION = "winning_solution"
    DATASET = "dataset"
    MODEL = "model"  # architecture / checkpoint refs


class ResearchArtifact(BaseModel):
    """Universal research object — one schema for every source kind."""

    id: str  # stable: paper:…, exp:14, repo:owner/name, …
    type: ResearchArtifactType
    source: str  # semantic_scholar | github | m2 | kaggle | reddit | user | …
    title: str = ""  # human label (also mirrored in metadata if useful)
    metadata: dict[str, Any] = Field(default_factory=dict)  # type-specific extras
    summary: str = ""  # short card — NOT a full-document TL;DR
    techniques: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)  # related artifact ids / evidence
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    competition_slug: str | None = None

    # Deprecated aliases during migration — prefer the fields above.
    # concepts → metadata/tags; evidence → references; payload → metadata.
    concepts: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchArtifacts(BaseModel):
    """One analyzer's emission — a bag of ResearchArtifact (+ soft-fail notes)."""

    analyzer: str
    items: list[ResearchArtifact] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)  # rollup convenience
    opportunities: list[str] = Field(default_factory=list)


class AnalyzeContext(BaseModel):
    """Everything an analyzer needs to run — no analyzer calls another analyzer."""

    competition: str  # normalized slug
    runs_dir: Path
    knowledge_dir: Path
    refresh: bool = False
    url: str | None = None  # original URL when a URL was passed instead of a slug

    @property
    def research_dir(self) -> Path:
        """Local (gitignored) research tree root for this competition."""
        return self.knowledge_dir / self.competition / "research"

    @property
    def reports_dir(self) -> Path:
        return self.research_dir / "reports"

    @property
    def report_path(self) -> Path:
        """Canonical analyze.json path (design §12.5)."""
        return self.reports_dir / "analyze.json"


class RetrievalSlice(BaseModel):
    """Placeholder retrieval section (filled by Plan 9)."""

    papers: list[str] = Field(default_factory=list)
    experiments: list[str] = Field(default_factory=list)
    repositories: list[str] = Field(default_factory=list)
    discussions: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class TechniqueBuckets(BaseModel):
    external_recommendations: list[str] = Field(default_factory=list)
    locally_validated: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    """Canonical ``analyze.json`` contract (design §12.5).

    Milestone 3 Plan 1 writes a stub: the envelope is stable, but most sections
    are empty until later plans (analyzers, knowledge hub, retrieval, hypothesis
    assistant) fill them.
    """

    schema_version: int = SCHEMA_VERSION
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    competition: dict[str, Any] = Field(default_factory=dict)
    analyzers: list[str] = Field(default_factory=list)
    artifacts: list[ResearchArtifact] = Field(default_factory=list)
    related_competitions: list[dict[str, Any]] = Field(default_factory=list)
    papers: list[dict[str, Any]] = Field(default_factory=list)
    repositories: list[dict[str, Any]] = Field(default_factory=list)
    transfer_opportunities: list[dict[str, Any]] = Field(default_factory=list)
    forum_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    knowledge_units: list[dict[str, Any]] = Field(default_factory=list)
    retrieval: RetrievalSlice = Field(default_factory=RetrievalSlice)
    hypothesis_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    techniques: TechniqueBuckets = Field(default_factory=TechniqueBuckets)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    suggested_experiments: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
