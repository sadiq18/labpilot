"""``ExperimentReviewerAgent`` — diagnose an experiment outcome.

The comparator (CV/LB deltas) stays deterministic (design §2.4 "No"); this
agent only interprets those numbers. Emits an :class:`ExperimentReview`.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
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

    def _run_rule_engine(self, context: StructuredContext) -> ExperimentReview:
        d = context.data
        cv = _as_float(d.get("cv_delta"))
        lb = _as_float(d.get("lb_delta"))
        changes = coerce_str_list(d.get("changes"))

        if cv is not None and lb is not None and cv > 0 and lb < 0:
            diagnosis = (
                "CV improved but LB regressed: likely validation/test "
                "distribution mismatch or overfitting to the CV split."
            )
        elif cv is not None and cv < 0:
            diagnosis = "CV regressed; the change did not help on local validation."
        elif cv is not None and cv > 0:
            diagnosis = "CV improved; change looks promising pending LB confirmation."
        else:
            diagnosis = "Insufficient metric signal to diagnose the outcome."

        suggestions = [f"Re-examine effect of: {c}" for c in changes]
        return ExperimentReview(diagnosis=diagnosis, suggestions=suggestions, confidence=0.5)
