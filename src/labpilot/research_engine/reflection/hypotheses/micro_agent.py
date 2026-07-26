"""HypothesisRevisionAgent — status rationale / revision prose."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext


class HypothesisRevisionDraft(BaseModel):
    outcome: str = "inconclusive"  # confirmed|rejected|partial|inconclusive
    why: str = ""
    revised_prediction: str = ""
    next_checks: list[str] = Field(default_factory=list)


class HypothesisRevisionAgent(BaseMicroAgent):
    name = "HypothesisRevisionAgent"
    output_model = HypothesisRevisionDraft

    def system_prompt(self) -> str:
        return (
            "Revise a hypothesis given experiment evidence and critic signals. "
            "Respond ONLY with JSON: "
            '{"outcome": "confirmed"|"rejected"|"partial"|"inconclusive", '
            '"why": str, "revised_prediction": str, "next_checks": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Hypothesis + evidence:\n{context.data}\nNotes:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> HypothesisRevisionDraft:
        d = context.data
        outcome = str(d.get("hypothesis_outcome") or "inconclusive")
        cause = str(d.get("likely_cause") or "Insufficient signal to revise firmly.")
        prediction = str(d.get("prediction") or "")
        why = cause
        if outcome == "partial":
            why = f"Partial support: {cause}"
        revised = prediction
        if outcome == "rejected" and prediction:
            revised = f"Revisit: {prediction}"
        checks = []
        if outcome in {"inconclusive", "partial"}:
            checks.append("Run a higher-powered follow-up with clearer controls.")
        elif outcome == "confirmed":
            checks.append("Promote to a research claim if evidence stays strong.")
        else:
            checks.append("Do not retry the same change without a new mechanism.")
        return HypothesisRevisionDraft(
            outcome=outcome,
            why=why,
            revised_prediction=revised,
            next_checks=checks,
        )
