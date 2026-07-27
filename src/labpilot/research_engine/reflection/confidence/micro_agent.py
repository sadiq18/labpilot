"""ConfidenceEstimatorAgent — qualitative confidence in the critic verdict."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class ConfidenceEstimate(BaseModel):
    label: str = "medium"  # high | medium | low
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class ConfidenceEstimatorAgent(BaseMicroAgent):
    name = "ConfidenceEstimatorAgent"
    output_model = ConfidenceEstimate

    def system_prompt(self) -> str:
        return (
            "Estimate qualitative confidence in the experiment verdict given "
            "evidence strength and metric deltas. Respond ONLY with JSON: "
            '{"label": "high"|"medium"|"low", "score": float, "rationale": str}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Signals:\n{context.data}\nNotes:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> ConfidenceEstimate:
        d = context.data
        strength = str(d.get("strength") or "moderate")
        delta = _as_float(d.get("cv_delta"))
        if strength == "strong" or (delta is not None and abs(delta) >= 0.01):
            return ConfidenceEstimate(
                label="high",
                score=0.8,
                rationale="Clear metric movement beyond noise / strong evidence.",
            )
        if strength == "rejected" or strength == "weak":
            return ConfidenceEstimate(
                label="low",
                score=0.4,
                rationale="Weak or rejected evidence reduces confidence.",
            )
        return ConfidenceEstimate(
            label="medium",
            score=0.55,
            rationale="Moderate evidence; treat verdict as provisional.",
        )
