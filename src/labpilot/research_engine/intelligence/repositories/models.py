"""Normalized GitHub repository and transfer-opportunity models (Plan 7)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.feature_recipes import FeatureRecipe


class RepoCategory(StrEnum):
    WINNING_SOLUTION = "winning_solution"
    BASELINE = "baseline"
    DOMAIN_LIBRARY = "domain_library"
    TRAINING_PIPELINE = "training_pipeline"
    AUGMENTATION = "augmentation"
    OTHER = "other"


class RepoSearchQuery(BaseModel):
    category: RepoCategory
    query: str


class RepoSearchPlan(BaseModel):
    queries: list[RepoSearchQuery] = Field(default_factory=list)


class Repository(BaseModel):
    id: str
    full_name: str
    url: str
    description: str = ""
    stars: int | None = None
    topics: list[str] = Field(default_factory=list)
    categories: list[RepoCategory] = Field(default_factory=list)
    primary_category: RepoCategory = RepoCategory.OTHER
    readme_excerpt: str = ""
    key_files: list[str] = Field(default_factory=list)
    file_texts: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    language: str | None = None
    relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    linked_paper_ids: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RepoKnowledge(BaseModel):
    """Transferable ML engineering knowledge, never a README summary."""

    repo_id: str = ""
    full_name: str = ""
    architecture: list[str] = Field(default_factory=list)
    loss: list[str] = Field(default_factory=list)
    augmentation: list[str] = Field(default_factory=list)
    training_tricks: list[str] = Field(default_factory=list)
    interesting_files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    feature_recipes: list[FeatureRecipe] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    grounded_in: Literal["readme", "code_excerpt", "deps", "mixed"] = "mixed"


class LocalCodeProfile(BaseModel):
    architecture: list[str] = Field(default_factory=list)
    loss: list[str] = Field(default_factory=list)
    augmentation: list[str] = Field(default_factory=list)
    training_tricks: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    files_scanned: list[str] = Field(default_factory=list)


class EffortEstimate(StrEnum):
    MINUTES_5 = "5m"
    MINUTES_20 = "20m"
    HOURS_1 = "1h"
    HOURS_4 = "4h"
    DAYS = "days"
    UNKNOWN = "unknown"


class ExpectedGain(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class TransferOpportunity(BaseModel):
    repo_id: str
    summary: str
    deltas: list[str] = Field(default_factory=list)
    local_baseline: str | None = None
    remote_choice: str | None = None
    effort: EffortEstimate = EffortEstimate.UNKNOWN
    expected_gain: ExpectedGain = ExpectedGain.UNKNOWN
    interesting_files: list[str] = Field(default_factory=list)
    hypothesis_hint: str | None = None
