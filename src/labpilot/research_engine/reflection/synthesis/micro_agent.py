"""EvidenceSynthesisAgent — Current Understanding narrative rollup."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list


class EvidenceSynthesisDraft(BaseModel):
    summary: str = ""
    open_questions: list[str] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)


class EvidenceSynthesisAgent(BaseMicroAgent):
    name = "EvidenceSynthesisAgent"
    output_model = EvidenceSynthesisDraft

    def system_prompt(self) -> str:
        return (
            "Synthesize the competition's current research understanding from "
            "evidence buckets, beliefs, and open hypotheses. Respond ONLY with JSON: "
            '{"summary": str, "open_questions": [str], "key_takeaways": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"State:\n{context.data}\nNotes:\n{context.text}"
