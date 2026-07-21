"""``HypothesisGeneratorAgent`` — draft a testable hypothesis from evidence.

Creative reasoning over already-retrieved evidence (design §2.4 "Yes"). Emits a
:class:`HypothesisDraft`; the caller persists it into the M2 Hypothesis store.
No autonomous planning or execution.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.micro_agents.artifacts import HypothesisDraft


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class HypothesisGeneratorAgent(BaseMicroAgent):
    name = "HypothesisGeneratorAgent"
    output_model = HypothesisDraft

    def system_prompt(self) -> str:
        return (
            "You draft ONE testable ML experiment hypothesis from the given "
            'evidence. Respond ONLY with JSON: {"observation": str, '
            '"prediction": str, "rationale": str, "expected_impact": float, '
            '"confidence": float}. Ground every field in the evidence; do not '
            "invent results."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Question: {context.question}\nEvidence:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> HypothesisDraft:
        d = context.data
        observation = str(d.get("observation") or context.question or "")
        return HypothesisDraft(
            observation=observation,
            prediction=str(d.get("prediction", "")),
            rationale=str(d.get("rationale", "")),
            expected_impact=_as_float(d.get("expected_impact"), 0.0),
            confidence=_as_float(d.get("confidence"), 0.5),
        )
