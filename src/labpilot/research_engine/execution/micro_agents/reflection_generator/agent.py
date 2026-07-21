"""``ReflectionGeneratorAgent`` — diagnose a finished run for `reflection.json`.

Execution-side reasoning slice (design §2.4 "Yes" — Reflection). Deterministic
comparator inputs in, structured diagnosis out. No auto-run: it only drafts a
reflection; the caller persists it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list


class ReflectionDraft(BaseModel):
    """Structured reflection fields (maps to the M2 `reflection.json` shape)."""

    summary: str = ""
    likely_cause: str = ""
    next_steps: list[str] = Field(default_factory=list)


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class ReflectionGeneratorAgent(BaseMicroAgent):
    name = "ReflectionGeneratorAgent"
    output_model = ReflectionDraft

    def system_prompt(self) -> str:
        return (
            "You reflect on a completed ML experiment run given deterministic "
            "CV/LB deltas and the changes applied. Respond ONLY with JSON: "
            '{"summary": str, "likely_cause": str, "next_steps": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Run signals:\n{context.data}\nNotes:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> ReflectionDraft:
        d = context.data
        cv = _as_float(d.get("cv_delta"))
        lb = _as_float(d.get("lb_delta"))
        changes = coerce_str_list(d.get("changes"))

        parts: list[str] = []
        if cv is not None:
            parts.append(f"CV delta {cv:+.4f}")
        if lb is not None:
            parts.append(f"LB delta {lb:+.4f}")
        summary = "; ".join(parts) or "No metric deltas available."

        if cv is not None and lb is not None and cv > 0 > lb:
            likely_cause = "Validation/test distribution mismatch or CV overfitting."
        elif cv is not None and cv < 0:
            likely_cause = "Change did not help local validation."
        else:
            likely_cause = "Inconclusive from available signals."

        next_steps = [f"Investigate: {c}" for c in changes]
        return ReflectionDraft(summary=summary, likely_cause=likely_cause, next_steps=next_steps)
