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
