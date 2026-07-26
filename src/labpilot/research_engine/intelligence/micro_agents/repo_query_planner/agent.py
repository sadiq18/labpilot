"""Optional LLM upgrade for typed, category-aware GitHub search queries."""

from __future__ import annotations

import json

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.repositories.models import (
    RepoSearchPlan,
    RepoSearchQuery,
)


class RepoQueryPlannerAgent(BaseMicroAgent):
    name = "RepoQueryPlannerAgent"
    output_model = RepoSearchPlan

    def system_prompt(self) -> str:
        return (
            "Build GitHub repository search queries for an ML competition. "
            "Respond ONLY with JSON: {\"queries\":[{\"category\":"
            "\"winning_solution|baseline|domain_library|training_pipeline|"
            "augmentation|other\",\"query\":\"...\"}]}. "
            "Rules: keep at most 8 queries; prefer 3-6 short tokens; "
            "use at most ONE quoted phrase per query; include the competition "
            "slug words or 2-3 core keywords; language:Python is allowed. "
            "Avoid stacking many required words like competition+solution+baseline. "
            "Never summarize the competition or call GitHub."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        return (
            f"Competition: {context.competition}\n"
            f"Context: {context.text}\n"
            f"Deterministic seed: {json.dumps(context.data.get('seed_queries', []))}\n"
            "Prefer broad recall over ultra-precise phrase matches."
        )

    def _run_rule_engine(self, context: StructuredContext) -> RepoSearchPlan:
        raw = context.data.get("seed_queries") or []
        queries: list[RepoSearchQuery] = []
        for item in raw:
            try:
                queries.append(RepoSearchQuery.model_validate(item))
            except (TypeError, ValueError):
                continue
        return RepoSearchPlan(queries=queries[:8])
