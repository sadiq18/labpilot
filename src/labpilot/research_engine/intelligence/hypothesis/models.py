"""Hypothesis Assistant models — recommendation cards only (design §10)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.shared.experiments.models import (
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
    STACKED = "stacked"
    COMBINATION = "combination"
    UNUSED_BELIEF = "unused_belief"
    UNUSED_CLAIM = "unused_claim"
    ABLATION = "ablation"


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
    parent_hypothesis_id: str | None = None
    technique_stack: list[str] = Field(default_factory=list)
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
    #: Numeric estimate of the metric delta (e.g. 0.015) — LLM draft when
    #: available, otherwise mapped from the qualitative ``expected_impact``.
    expected_impact_value: float = 0.0
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
    technique: str = ""
    parent_hypothesis_id: str | None = None
    technique_stack: list[str] = Field(default_factory=list)
    combo_techniques: list[str] = Field(default_factory=list)
    combo_rationale: str = ""


class HypothesisAssistantResult(BaseModel):
    recommendations: list[HypothesisRecommendation] = Field(default_factory=list)
    #: Hypotheses newly created by this run (0 when everything was already covered).
    new_count: int = 0
    notes: list[str] = Field(default_factory=list)
    context: ResearchContext | None = None
