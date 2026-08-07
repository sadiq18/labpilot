"""RecommendationAgent — next-experiment suggestion prose + action."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list


class RecommendationDraft(BaseModel):
    action: str = "analyze_or_hypothesize"
    rationale: str = ""
    command: str = ""
    candidates: list[str] = Field(default_factory=list)


class RecommendationAgent(BaseMicroAgent):
    name = "RecommendationAgent"
    output_model = RecommendationDraft

    def system_prompt(self) -> str:
        return (
            "Recommend the next experiment for this competition. Respond ONLY with JSON: "
            '{"action": str, "rationale": str, "command": str, "candidates": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Journal state:\n{context.data}\nNotes:\n{context.text}"
