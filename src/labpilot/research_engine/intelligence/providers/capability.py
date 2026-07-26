"""Typed capability envelopes for Competition Intelligence (design §3.5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.competition.models import MetricSpec
from labpilot.research_engine.intelligence.models import ResearchArtifact


class CapabilityResult(BaseModel):
    """Explicit availability — never silent empty lists for unsupported capabilities."""

    available: bool
    status: Literal["ok", "unavailable", "error"] = "unavailable"
    reason: str = ""
    items: list[ResearchArtifact] = Field(default_factory=list)


class RelatedCompetition(BaseModel):
    slug: str
    title: str = ""
    relation: Literal[
        "previous_edition",
        "similar_domain",
        "similar_metric",
        "similar_modality",
        "other",
    ]
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    rationale: str = ""
    tags_overlap: list[str] = Field(default_factory=list)


class ExternalDataPolicy(BaseModel):
    status: Literal["ok", "unavailable", "error"] = "unavailable"
    allowed: bool | None = None  # None if unknown
    pretrained_weights: bool | None = None
    notes: str = ""


class InferenceLimits(BaseModel):
    status: Literal["ok", "unavailable", "error"] = "unavailable"
    runtime_notes: str = ""
    hardware_notes: str = ""
    internet_allowed: bool | None = None
    notes: str = ""


class CompetitionProfile(BaseModel):
    """Kaggle-expert brief — canonical competition section of analyze.json."""

    slug: str
    title: str = ""
    url: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    metric: MetricSpec | None = None
    problem_type: str | None = None
    rules_excerpt: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    timeline: dict[str, Any] = Field(default_factory=dict)
    submission: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    external_data: ExternalDataPolicy = Field(default_factory=ExternalDataPolicy)
    inference_limits: InferenceLimits = Field(default_factory=InferenceLimits)
    dataset_catalog: CapabilityResult | None = None
    leaderboard: CapabilityResult | None = None
    winning_solutions: CapabilityResult | None = None
    previous_editions: list[RelatedCompetition] = Field(default_factory=list)
    related_competitions: list[RelatedCompetition] = Field(default_factory=list)
    capability_notes: list[str] = Field(default_factory=list)
    page_enrichment_source: str = ""  # llm | rule_engine | unavailable
