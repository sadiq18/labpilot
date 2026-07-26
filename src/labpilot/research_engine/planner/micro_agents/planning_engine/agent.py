"""``ResearchPlannerAgent`` — the single Planning Engine LLM stage.

Option B: the compiler always builds a rule_engine template baseline first,
then this agent optionally revises it into a slim :class:`ResearchPlanDraft`.
``_run_rule_engine`` returns the baseline unchanged (identity pass-through).
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.planner.schemas.draft import ResearchPlanDraft

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class ResearchPlannerAgent(BaseMicroAgent):
    name = "ResearchPlannerAgent"
    output_model = ResearchPlanDraft

    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client)
        self._prompts_dir = _PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(self._prompts_dir),
            autoescape=select_autoescape(default=False),
        )

    def system_prompt(self) -> str:
        return (self._prompts_dir / "planning_engine_system.md").read_text()

    def user_prompt(self, context: StructuredContext) -> str:
        data = context.data
        baseline = data.get("baseline_draft") or {}
        baseline_json = json.dumps(baseline, indent=2, ensure_ascii=False)
        return self._env.get_template("planning_engine_user.j2").render(
            competition=context.competition,
            hypothesis_id=str(data.get("hypothesis_id") or ""),
            observation=str(data.get("observation") or ""),
            reason=str(data.get("reason") or ""),
            prediction=str(data.get("prediction") or ""),
            expected_impact=data.get("expected_impact", 0.0),
            confidence=data.get("confidence", 0.0),
            tags=list(data.get("tags") or []),
            goal=str(data.get("goal") or context.question or ""),
            current_state=str(data.get("current_state") or ""),
            expected_outcome=str(data.get("expected_outcome") or ""),
            technique_names=list(data.get("technique_names") or []),
            belief_summaries=list(data.get("belief_summaries") or []),
            brief_excerpt=str(data.get("brief_excerpt") or context.text or ""),
            baseline_json=baseline_json,
        )

    def _run_rule_engine(self, context: StructuredContext) -> ResearchPlanDraft:
        """Identity: return the compiler's baseline draft unchanged."""
        baseline = context.data.get("baseline_draft")
        if isinstance(baseline, ResearchPlanDraft):
            return baseline
        if isinstance(baseline, dict):
            return ResearchPlanDraft.model_validate(baseline)
        return ResearchPlanDraft(
            goal=str(context.data.get("goal") or context.question or ""),
            current_state=str(context.data.get("current_state") or ""),
            expected_outcome=str(context.data.get("expected_outcome") or ""),
            tasks=[],
        )
