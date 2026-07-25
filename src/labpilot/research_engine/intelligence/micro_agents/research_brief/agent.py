"""``ResearchBriefAgent`` — narrative slices for the Research Brief."""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.brief.models import ResearchBriefNarrative


class ResearchBriefAgent(BaseMicroAgent):
    name = "ResearchBriefAgent"
    output_model = ResearchBriefNarrative

    def system_prompt(self) -> str:
        return (
            "You write a concise researcher briefing before ML experimentation. "
            "Respond ONLY with JSON: "
            '{"problem_summary": str, "key_risks": [str], "recommended_focus": str}. '
            "Ground every claim in the provided structured evidence; do not invent scores."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return (
            f"Competition: {context.competition}\n"
            f"Question: {context.question}\n"
            f"Evidence:\n{context.text}"
        )

    def _run_rule_engine(self, context: StructuredContext) -> ResearchBriefNarrative:
        d = context.data
        title = str(d.get("title") or context.competition or "Competition")
        problem_type = str(d.get("problem_type") or "unknown problem")
        dataset = str(d.get("dataset_overview") or "").strip()
        rules = str(d.get("rules_and_metric") or "").strip()
        bits = [f"{title} is a {problem_type} competition"]
        if rules:
            bits.append(rules.split("|")[0].strip())
        if dataset:
            bits.append(dataset)
        problem_summary = ". ".join(bits) + "."

        risks = coerce_str_list(d.get("known_risks"))
        hypotheses = coerce_str_list(d.get("top_hypotheses"))
        suggested = coerce_str_list(d.get("suggested_experiments"))
        focus = ""
        if suggested:
            focus = suggested[0]
        elif hypotheses:
            focus = hypotheses[0]
        elif coerce_str_list(d.get("winning_techniques")):
            focus = f"Validate {coerce_str_list(d.get('winning_techniques'))[0]}"

        return ResearchBriefNarrative(
            problem_summary=problem_summary,
            key_risks=risks[:8],
            recommended_focus=focus,
        )
