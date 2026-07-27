"""Next-experiment recommendations via Recommendation Micro Agent."""

from __future__ import annotations

from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.reflection.recommendation.micro_agent import (
    RecommendationAgent,
    RecommendationDraft,
)


def recommend_next_experiment(
    understanding: dict[str, Any],
    *,
    assessment_recommendation: str = "",
    llm_client: Any | None = None,
) -> dict[str, Any]:
    agent = RecommendationAgent(llm_client=llm_client)
    open_ids = [h.get("id") for h in (understanding.get("open_hypotheses") or []) if h.get("id")]
    draft = agent.run(
        StructuredContext(
            competition=str(understanding.get("competition") or ""),
            data={
                "open_hypothesis_ids": open_ids,
                "evidence_by_strength": understanding.get("evidence_by_strength") or {},
                "assessment_recommendation": assessment_recommendation,
            },
        )
    )
    assert isinstance(draft, RecommendationDraft)
    return {
        "action": draft.action,
        "rationale": draft.rationale,
        "command": draft.command,
        "candidates": draft.candidates,
        "generated_by": "llm" if agent.last_used_llm else "rule_engine",
    }
