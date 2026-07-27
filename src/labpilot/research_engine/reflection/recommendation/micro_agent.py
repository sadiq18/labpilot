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

    def _run_rule_engine(self, context: StructuredContext) -> RecommendationDraft:
        d = context.data
        open_hyps = coerce_str_list(d.get("open_hypothesis_ids"))
        evidence = d.get("evidence_by_strength") or {}
        critic_rec = str(d.get("assessment_recommendation") or "")
        if open_hyps:
            hid = open_hyps[0]
            return RecommendationDraft(
                action="plan_create",
                rationale=f"Test open hypothesis {hid}.",
                command=f"research plan create --hypothesis {hid}",
                candidates=open_hyps[:3],
            )
        rejected = int(evidence.get("rejected") or 0)
        strong = int(evidence.get("strong") or 0)
        if rejected > strong:
            return RecommendationDraft(
                action="baseline_recheck",
                rationale="More rejected than strong evidence — re-verify P-001 baseline.",
                command="research plan create <slug> --baseline",
            )
        if critic_rec:
            return RecommendationDraft(
                action="follow_critic",
                rationale=critic_rec,
                command="research plan create <slug> --hypothesis <H-xxx>",
            )
        return RecommendationDraft(
            action="analyze_or_hypothesize",
            rationale="No open hypotheses — analyze/hypothesize before the next plan.",
            command="research hypothesize list --competition <slug>",
        )
