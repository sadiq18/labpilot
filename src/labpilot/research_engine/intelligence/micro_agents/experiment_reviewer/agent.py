"""``ExperimentReviewerAgent`` — diagnose an experiment outcome.

The comparator (CV/LB deltas) stays deterministic (design §2.4 "No"); this
agent only interprets those numbers. Emits an :class:`ExperimentReview`.
"""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.micro_agents.artifacts import ExperimentReview


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class ExperimentReviewerAgent(BaseMicroAgent):
    name = "ExperimentReviewerAgent"
    output_model = ExperimentReview

    def system_prompt(self) -> str:
        return (
            "You diagnose an ML experiment given deterministic CV/LB metric "
            "deltas and the changes applied. Respond ONLY with JSON: "
            '{"diagnosis": str, "suggestions": [str], "confidence": float}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Experiment signals:\n{context.data}\nNotes:\n{context.text}"
