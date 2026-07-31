"""``ComboPortfolioAgent`` — pick complementary technique merges from a shortlist."""

from __future__ import annotations

import json

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.intelligence.hypothesis.combo import rule_engine_pick_combos
from labpilot.research_engine.intelligence.micro_agents.artifacts import (
    ComboPick,
    ComboPortfolioDraft,
)


class ComboPortfolioAgent(BaseMicroAgent):
    name = "ComboPortfolioAgent"
    output_model = ComboPortfolioDraft

    def system_prompt(self) -> str:
        return (
            "You choose the best small combinations of ML techniques to try "
            "together in ONE experiment (size 2, optionally 3). "
            "Respond ONLY with JSON: "
            '{"picks":[{"techniques":[str],"rationale":str,"confidence":float,'
            '"expected_impact":float}]}. '
            "Only use techniques that appear in the provided shortlist portfolios. "
            "Prefer complementary categories (e.g. feature engineering + model) "
            "over two conflicting models. Return at most 3 picks. "
            "Do not invent techniques."
        )

    def user_prompt(self, context: StructuredContext) -> str:
        data = context.data
        return (
            f"Competition: {context.competition}\n"
            f"Parent stack: {data.get('parent_stack')}\n"
            f"Parent metrics: {data.get('parent_metrics')}\n"
            f"Avoid pairs: {data.get('avoid_pairs')}\n"
            f"Failed techniques: {data.get('failed')}\n"
            f"Brief:\n{context.text}\n\n"
            f"Shortlist portfolios (JSON):\n"
            f"{json.dumps(data.get('shortlist') or [], ensure_ascii=False)[:8000]}\n"
            f"Pick up to {int(data.get('limit') or 3)} best merges."
        )

    def _run_rule_engine(self, context: StructuredContext) -> ComboPortfolioDraft:
        shortlist = list(context.data.get("shortlist") or [])
        limit = int(context.data.get("limit") or 3)
        picks_raw = rule_engine_pick_combos(shortlist, limit=limit)
        return ComboPortfolioDraft(
            picks=[
                ComboPick(
                    techniques=list(p.get("techniques") or []),
                    rationale=str(p.get("rationale") or ""),
                    confidence=float(p.get("confidence") or 0.6),
                    expected_impact=float(p.get("expected_impact") or 0.015),
                )
                for p in picks_raw
            ]
        )
