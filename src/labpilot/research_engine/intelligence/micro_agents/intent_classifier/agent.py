"""``IntentClassifierAgent`` — classify a free-text research question into RetrievalIntent.

Classify-only: never invents answers or techniques. Rules path fills intent from
competition profile / pipeline hints when no LLM is available.
"""

from __future__ import annotations

from labpilot.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.retrieval.models import RetrievalIntent


class IntentClassifierAgent(BaseMicroAgent):
    name = "IntentClassifierAgent"
    output_model = RetrievalIntent

    def system_prompt(self) -> str:
        return (
            "You classify a research question into a RetrievalIntent JSON object. "
            "Respond ONLY with JSON matching the schema. Do NOT invent techniques, "
            "papers, or experiment recommendations — only fill intent fields "
            "(task, dataset, domain, goal, metric, query_type, need_* flags, "
            "current_pipeline, question). Prefer query_type values: "
            "hypothesis_generation, structured_query, explain, compare."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        question = context.text or "\n".join(context.items)
        profile = context.data.get("profile") or {}
        pipeline = context.data.get("pipeline") or []
        return (
            f"Question:\n{question}\n\n"
            f"Competition profile hints:\n{profile}\n\n"
            f"Current pipeline techniques:\n{pipeline}\n"
        )

    def _run_rule_engine(self, context: StructuredContext) -> RetrievalIntent:
        from labpilot.research_engine.intelligence.retrieval.intent import (
            classify_intent_rules,
        )

        return classify_intent_rules(
            question=context.text or "\n".join(context.items),
            profile=dict(context.data.get("profile") or {}),
            pipeline=list(context.data.get("pipeline") or []),
            query_type=context.data.get("query_type"),
        )
