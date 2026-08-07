"""RootCauseAgent — diagnose why an experiment outcome occurred."""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list


class ReflectionDraft(BaseModel):
    """Root-cause / critic draft (legacy reflection.json shape)."""

    summary: str = ""
    likely_cause: str = ""
    next_steps: list[str] = Field(default_factory=list)


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class RootCauseAgent(BaseMicroAgent):
    """Root-cause analysis for a finished experiment (LLM + rule_engine)."""

    name = "RootCauseAgent"
    output_model = ReflectionDraft

    def system_prompt(self) -> str:
        return (
            "You perform root-cause analysis on a completed ML experiment given "
            "deterministic metric deltas and config changes. Respond ONLY with JSON: "
            '{"summary": str, "likely_cause": str, "next_steps": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Run signals:\n{context.data}\nNotes:\n{context.text}"


# Compat: prior name used by execution registry / tests.
ReflectionGeneratorAgent = RootCauseAgent
