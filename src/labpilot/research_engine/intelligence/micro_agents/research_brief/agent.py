"""``ResearchBriefAgent`` — narrative slices for the Research Brief."""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.brief.models import ResearchBriefNarrative


class ResearchBriefAgent(BaseMicroAgent):
    name = "ResearchBriefAgent"
    output_model = ResearchBriefNarrative

    def system_prompt(self) -> str:
        return (
            "You write a concise researcher briefing before ML experimentation. "
            "Respond ONLY with JSON: "
            '{"problem_summary": str, "key_risks": [str], "recommended_focus": str}. '
            "Ground every claim in the provided structured evidence; do not invent scores. "
            "Write complete sentences that end with '.', '!', or '?'. "
            "Never end any sentence with '...' or '…'; prefer a shorter finished "
            "sentence over a truncated one."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return (
            f"Competition: {context.competition}\n"
            f"Question: {context.question}\n"
            f"Evidence:\n{context.text}"
        )
