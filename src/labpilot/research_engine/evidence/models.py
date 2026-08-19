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

    @property
    def delta_flags(self) -> list[str]:
        """Consistency flags the write-code checks raised for this run."""
        recorded = (self.metadata or {}).get("delta_flags")
        return [str(flag) for flag in recorded] if isinstance(recorded, list) else []

    @property
    def unverified_steps(self) -> list[str]:
        """Steps in this run whose evidence says they verified nothing."""
        recorded = (self.metadata or {}).get("unverified_steps")
        return [str(step) for step in recorded] if isinstance(recorded, list) else []

    @property
    def uncomparable_reason(self) -> str | None:
        """Why this card's two readings are not a comparison, if they are not.

        Set when the control and the treatment are the same execution, or when
        the two runs returned identical metrics. Every writer that re-derives
        `decision` — `submit_learn` when leaderboard results land, `repair` when
        a direction is corrected — must fold this into `missing_control`, or it
        will sign a verdict on a measurement that never varied. Carried in
        `metadata` rather than `decision_reason` because those writers overwrite
        the reason and would drop it.
        """
        recorded = (self.metadata or {}).get("uncomparable_reason")
        return str(recorded) if recorded else None

    @property
    def decision_summary(self) -> str:
        """`decision_reason`, qualified by the flags that should temper it.

        Derived rather than stored, and that is the whole point. The first
        version appended the flags to `decision_reason` when the card was
        built — and three separate writers recompute that field afterwards
        (`submit_learn` when leaderboard results land, `repair` when a
        direction is corrected, twice), each overwriting the text wholesale.
        The flags vanished precisely when a hypothesis reached the
        leaderboard, which is the confirmed case they matter most for.
        Reported on PR #119.

        `metadata["delta_flags"]` survives every one of those updates because
        none of them touch the key. Deriving from it means no future writer can
        lose the qualification by rewriting a sentence.
        """
        parts = [self.decision_reason]
        uncomparable = self.uncomparable_reason
        if uncomparable:
            # Without this the qualification is stored and never shown. Once
            # `submit_learn` recomputes, `decision_reason` is the bare
            # "missing_control" while `control_experiment` names a real
            # execution — a card contradicting itself, with the reason sitting
            # unread in `metadata`. Exactly the failure the `delta_flags`
            # treatment below was written for.
            parts.append(f"not a comparison: {uncomparable}")
        flags = self.delta_flags
        if flags:
            parts.append(f"{len(flags)} delta flag(s): {'; '.join(flags)}")
        unverified = self.unverified_steps
        if unverified:
            # A conclusion drawn from a run whose unit-test step skipped because
            # there were no tests is weaker than one where the tests passed, and
            # this is where that has to be visible. The stamp existed on the task
            # evidence and nothing read it — the same shape as `delta_flags`
            # sitting in a file no part of the system opened.
            parts.append(f"{len(unverified)} step(s) verified nothing: {', '.join(unverified)}")
        # No `lstrip` — the filter here is what handles an empty
        # `decision_reason`, and leaving a strip beside it invited someone to
        # remove the filter believing the strip covered it. Reported reviewing
        # PR #121.
        return " · ".join(part for part in parts if part)

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
