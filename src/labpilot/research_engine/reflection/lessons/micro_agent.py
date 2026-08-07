"""LessonGeneratorAgent — durable takeaway prose from an experiment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext


class LessonDraft(BaseModel):
    summary: str = ""
    category: str = "process"  # technique | process | pitfall
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class LessonGeneratorAgent(BaseMicroAgent):
    name = "LessonGeneratorAgent"
    output_model = LessonDraft

    def system_prompt(self) -> str:
        return (
            "Write one durable research lesson from this experiment outcome. "
            "Respond ONLY with JSON: "
            '{"summary": str, "category": "technique"|"process"|"pitfall", '
            '"confidence": float}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Outcome:\n{context.data}\nNotes:\n{context.text}"
