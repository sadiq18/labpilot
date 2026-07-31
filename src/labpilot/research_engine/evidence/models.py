"""Evidence Card — atomic learning unit for one hypothesis execution."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class StabilityOutcome(StrEnum):
    IMPROVED = "improved"
    SIMILAR = "similar"
    WORSE = "worse"
    UNKNOWN = "unknown"


class EvidenceDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ClaimEvidenceKind(StrEnum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class ExpectedOutcomes(BaseModel):
    cv_gain: float | None = None
    runtime: float | None = None  # fractional change vs control


class ObservedOutcomes(BaseModel):
    cv_gain: float | None = None
    lb_gain: float | None = None
    runtime: float | None = None
    stability: StabilityOutcome = StabilityOutcome.UNKNOWN
    memory_mb_delta: float | None = None
    parent_cv: float | None = None
    treatment_cv: float | None = None
    parent_cv_std: float | None = None
    treatment_cv_std: float | None = None
    train_time_s: float | None = None
    inference_time_s: float | None = None
    peak_memory_mb: float | None = None


class ClaimUpdate(BaseModel):
    claim: str
    evidence: ClaimEvidenceKind = ClaimEvidenceKind.NEUTRAL
    confidence_delta: float = 0.0
    technique: str = ""


class EvidenceCard(BaseModel):
    """Causal evidence for one treatment execution vs a control."""

    id: str = ""
    competition: str = ""
    hypothesis_id: str | None = None
    control_experiment: str | None = None
    treatment_experiment: str = ""
    control_hypothesis_id: str | None = None
    plan_id: str | None = None

    expected: ExpectedOutcomes = Field(default_factory=ExpectedOutcomes)
    observed: ObservedOutcomes = Field(default_factory=ObservedOutcomes)

    technique_attribution: dict[str, float] = Field(default_factory=dict)
    claim_updates: list[ClaimUpdate] = Field(default_factory=list)

    decision: EvidenceDecision = EvidenceDecision.INCONCLUSIVE
    decision_reason: str = ""
    reusable_for: list[str] = Field(default_factory=list)

    impact_error: float | None = None  # observed.cv_gain - expected.cv_gain
    maximize: bool = True
    noise_epsilon: float = 0.001
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_comparison_dict(self) -> dict[str, Any]:
        """Keys outcome._learning_deltas expects on comparison.json."""
        cv = self.observed.cv_gain
        return {
            "plan_id": self.plan_id,
            "execution_id": self.treatment_experiment,
            "compare_to": self.control_experiment or "missing_control",
            "control_experiment": self.control_experiment,
            "hypothesis_id": self.hypothesis_id,
            "evidence_card_id": self.id,
            "metrics": {
                "parent_cv": self.observed.parent_cv,
                "treatment_cv": self.observed.treatment_cv,
            },
            "delta": cv,
            "cv_delta": cv,
            "primary_delta": cv,
            "runtime_delta_frac": self.observed.runtime,
            "stability": self.observed.stability.value,
            "decision": self.decision.value,
            "outcome": self.decision.value,
            "technique_attribution": dict(self.technique_attribution),
        }
