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

    def _run_rule_engine(self, context: StructuredContext) -> EvidenceSynthesisDraft:
        d = context.data
        by_strength = d.get("evidence_by_strength") or {}
        beliefs = coerce_str_list(d.get("belief_lines"))
        open_hyps = coerce_str_list(d.get("open_hypothesis_lines"))
        summary = (
            "Current understanding: "
            f"strong={by_strength.get('strong', 0)}, "
            f"rejected={by_strength.get('rejected', 0)}, "
            f"beliefs={len(beliefs)}, open_hypotheses={len(open_hyps)}."
        )
        takeaways = beliefs[:3] or ["No validated beliefs yet."]
        return EvidenceSynthesisDraft(
            summary=summary,
            open_questions=open_hyps[:5] or ["No open hypotheses."],
            key_takeaways=takeaways,
        )
