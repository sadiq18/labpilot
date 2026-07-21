"""``PaperAnalyzerAgent`` — structured extraction from a single paper.

Flagship "Yes" pattern (design §2.4). The LLM is an information extractor, not
a chatbot: never "summarize this paper", always structured JSON validated
against :class:`PaperExtract`.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext, coerce_str_list
from labpilot.research_engine.intelligence.micro_agents.artifacts import PaperExtract


class PaperAnalyzerAgent(BaseMicroAgent):
    name = "PaperAnalyzerAgent"
    output_model = PaperExtract

    def system_prompt(self) -> str:
        return (
            "You extract structured research knowledge from an ML paper. "
            "Respond ONLY with a JSON object matching this schema: "
            '{"techniques": [str], "models": [str], "datasets": [str], '
            '"limitations": [str], "hypotheses": [str], "claims": [str]}. '
            "Do not summarize; extract named techniques, models, datasets, "
            "stated limitations, ideas worth testing, and concrete claims."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return f"Paper text:\n{context.text}"

    def _run_rule_engine(self, context: StructuredContext) -> PaperExtract:
        d = context.data
        return PaperExtract(
            techniques=coerce_str_list(d.get("techniques")),
            models=coerce_str_list(d.get("models")),
            datasets=coerce_str_list(d.get("datasets")),
            limitations=coerce_str_list(d.get("limitations")),
            hypotheses=coerce_str_list(d.get("hypotheses")),
            claims=coerce_str_list(d.get("claims")),
        )
