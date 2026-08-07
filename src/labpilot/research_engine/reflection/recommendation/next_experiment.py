"""Next-experiment recommendations via Recommendation Micro Agent."""

from __future__ import annotations

from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
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
    draft = run_or_none(
        agent,
        StructuredContext(
            competition=str(understanding.get("competition") or ""),
            data={
                "open_hypothesis_ids": open_ids,
                "evidence_by_strength": understanding.get("evidence_by_strength") or {},
                "assessment_recommendation": assessment_recommendation,
            },
        ),
    )
    if draft is None:
        if open_ids:
            draft = RecommendationDraft(
                action="plan_create",
                rationale=f"Test open hypothesis {open_ids[0]}.",
                command=f"research plan create --hypothesis {open_ids[0]}",
                candidates=open_ids[:3],
            )
        else:
            draft = RecommendationDraft(
                action="analyze_or_hypothesize",
                rationale=assessment_recommendation
                or "No open hypotheses — analyze/hypothesize before the next plan.",
                command="research hypothesize list --competition <slug>",
            )
    elif not isinstance(draft, RecommendationDraft):
        draft = RecommendationDraft.model_validate(draft.model_dump())
    return {
        "action": draft.action,
        "rationale": draft.rationale,
        "command": draft.command,
        "candidates": draft.candidates,
        "generated_by": "llm" if agent.last_used_llm else "template_fallback",
    }
