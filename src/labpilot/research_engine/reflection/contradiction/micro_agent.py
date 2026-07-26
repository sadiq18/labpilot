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

    def _run_rule_engine(self, context: StructuredContext) -> ContradictionReport:
        d = context.data
        effect = str(d.get("belief_effect") or "")
        prior_effect = str(d.get("prior_belief_effect") or "")
        strength = str(d.get("strength") or "")
        prior_ids = coerce_str_list(d.get("prior_claim_ids"))
        evidence_id = str(d.get("evidence_id") or "")

        contradicts = (
            effect == "contradicts"
            or (prior_effect == "positive" and strength == "rejected")
            or (prior_effect == "negative" and strength == "strong")
        )

        if not contradicts:
            return ContradictionReport(
                has_contradiction=False,
                summary="No contradiction detected from available signals.",
            )

        ids = list(prior_ids)
        if evidence_id:
            ids.append(evidence_id)
        return ContradictionReport(
            has_contradiction=True,
            summary=(
                f"New evidence ({strength}) conflicts with prior belief effect "
                f"'{prior_effect or 'unknown'}'."
            ),
            conflicting_ids=ids,
            resolution_hint="Mark claim contested or run a decisive follow-up experiment.",
        )
