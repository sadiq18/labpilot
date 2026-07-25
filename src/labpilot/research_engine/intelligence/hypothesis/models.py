"""Hypothesis Assistant models — recommendation cards only (design §10)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from labpilot.experiments.models import (
    HypothesisCreatedBy,
    HypothesisEvidenceRef,
    HypothesisGenerator,
    HypothesisOrigin,
)
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
)
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext


class HypothesisCandidateKind(StrEnum):
    BELIEF = "belief"
    PIPELINE_DIFF = "pipeline_diff"
    TRANSFER = "transfer"
    FAILURE_FIX = "failure_fix"
    TECHNIQUE = "technique"


class HypothesisCandidate(BaseModel):
    """Pre-rank candidate — deterministic generation only."""

    key: str
    kind: HypothesisCandidateKind
    title: str
    observation: str = ""
    reason: str = ""
    prediction: str = ""
    technique: str = ""
    expected_impact: ExpectedGain = ExpectedGain.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    implementation_effort: EffortEstimate = EffortEstimate.UNKNOWN
    evidence: list[HypothesisEvidenceRef] = Field(default_factory=list)
    origins: list[HypothesisOrigin] = Field(default_factory=list)
    avoids_failure_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    score_hint: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HypothesisRecommendation(BaseModel):
    """One ranked suggestion — recommendation only, never an execution plan step."""

    rank: int
    hypothesis_id: str
    title: str
    observation: str = ""
    reason: str = ""
    prediction: str = ""
    expected_impact: ExpectedGain = ExpectedGain.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    supporting_evidence: list[HypothesisEvidenceRef] = Field(default_factory=list)
    implementation_effort: EffortEstimate = EffortEstimate.UNKNOWN
    origins: list[HypothesisOrigin] = Field(default_factory=list)
    avoids_failure_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    created_by: HypothesisCreatedBy = HypothesisCreatedBy.ANALYZE
    generator: HypothesisGenerator = HypothesisGenerator.RULE_ENGINE
    origin: HypothesisOrigin = HypothesisOrigin.MIXED
    tags: list[str] = Field(default_factory=list)


class HypothesisAssistantResult(BaseModel):
    recommendations: list[HypothesisRecommendation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    context: ResearchContext | None = None
