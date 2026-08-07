"""ContradictionDetectorAgent — narrative over conflicting evidence/beliefs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list


class ContradictionReport(BaseModel):
    has_contradiction: bool = False
    summary: str = ""
    conflicting_ids: list[str] = Field(default_factory=list)
    resolution_hint: str = ""


class ContradictionDetectorAgent(BaseMicroAgent):
    name = "ContradictionDetectorAgent"
    output_model = ContradictionReport

    def system_prompt(self) -> str:
        return (
            "Detect contradictions between new experiment evidence and prior "
            "beliefs/claims. Respond ONLY with JSON: "
            '{"has_contradiction": bool, "summary": str, "conflicting_ids": [str], '
            '"resolution_hint": str}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Context:\n{context.data}\nNotes:\n{context.text}"
