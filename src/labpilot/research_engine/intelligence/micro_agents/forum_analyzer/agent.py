"""``ForumAnalyzerAgent`` — mine structured signals from a discussion thread.

Discussion analysis is **not** a Phase 1 default (no provider yet), so this
agent's ``rule_engine`` path simply surfaces whatever pre-parsed signals the
caller supplies. Emits :class:`ForumExtract`.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.micro_agents.artifacts import ForumExtract


class ForumAnalyzerAgent(BaseMicroAgent):
    name = "ForumAnalyzerAgent"
    output_model = ForumExtract

    def system_prompt(self) -> str:
        return (
            "You extract actionable signals from a Kaggle discussion thread. "
            'Respond ONLY with JSON: {"mistakes": [str], "discoveries": [str], '
            '"dataset_bugs": [str], "lb_shakeups": [str], "ood_notes": [str]}.'
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Discussion text:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> ForumExtract:
        d = context.data
        return ForumExtract(
            mistakes=coerce_str_list(d.get("mistakes")),
            discoveries=coerce_str_list(d.get("discoveries")),
            dataset_bugs=coerce_str_list(d.get("dataset_bugs")),
            lb_shakeups=coerce_str_list(d.get("lb_shakeups")),
            ood_notes=coerce_str_list(d.get("ood_notes")),
        )
