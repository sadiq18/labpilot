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
