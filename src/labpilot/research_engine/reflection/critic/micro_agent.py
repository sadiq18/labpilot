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

    def _run_rule_engine(self, context: StructuredContext) -> ReflectionDraft:
        d = context.data
        cv = _as_float(d.get("cv_delta"))
        lb = _as_float(d.get("lb_delta"))
        changes = coerce_str_list(d.get("changes"))
        strength = str(d.get("strength") or "")

        parts: list[str] = []
        if cv is not None:
            parts.append(f"CV delta {cv:+.4f}")
        if lb is not None:
            parts.append(f"LB delta {lb:+.4f}")
        if strength:
            parts.append(f"evidence strength={strength}")
        summary = "; ".join(parts) or "No metric deltas available."

        if strength == "rejected" or (cv is not None and cv < 0):
            likely_cause = "Change did not help local validation (or run failed)."
        elif cv is not None and lb is not None and cv > 0 > lb:
            likely_cause = "Validation/test distribution mismatch or CV overfitting."
        elif strength == "strong" or (cv is not None and cv > 0):
            likely_cause = "Change improved the tracked metric within noise rules."
        else:
            likely_cause = "Inconclusive from available signals."

        next_steps = [f"Investigate: {c}" for c in changes] or [
            "Compare against P-001 baseline and propose the next hypothesis."
        ]
        return ReflectionDraft(
            summary=summary, likely_cause=likely_cause, next_steps=next_steps
        )


# Compat: prior name used by execution registry / tests.
ReflectionGeneratorAgent = RootCauseAgent
