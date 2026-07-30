"""``CodeEngineerAgent`` — propose full pipeline code as a typed CodeProposal.

Always generates from scratch from profile / data inventory. Never uses Jinja
template packs. Soft-fails to an empty proposal when the LLM is unavailable
(caller may apply a last-resort stub).
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.execution.schemas.code_proposal import CodeProposal

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class CodeEngineerAgent(BaseMicroAgent):
    name = "CodeEngineerAgent"
    output_model = CodeProposal

    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client)
        self._prompts_dir = _PROMPTS_DIR
        self._env = Environment(
            loader=FileSystemLoader(self._prompts_dir),
            autoescape=select_autoescape(default=False),
        )

    def system_prompt(self) -> str:
        return (self._prompts_dir / "code_engineer_system.md").read_text(encoding="utf-8")

    def user_prompt(self, context: StructuredContext) -> str:
        data = context.data
        profile = data.get("profile_summary") or {}
        choice = data.get("baseline_choice") or {}
        return self._env.get_template("code_engineer_user.j2").render(
            competition=context.competition,
            goal=context.question or str(data.get("plan_goal") or ""),
            brief_excerpt=context.text or str(data.get("brief_excerpt") or ""),
            task_id=str(data.get("task_id") or ""),
            task_type=str(data.get("task_type") or ""),
            task_description=str(data.get("task_description") or ""),
            plan_id=str(data.get("plan_id") or ""),
            plan_kind=str(data.get("plan_kind") or ""),
            hypothesis_id=str(data.get("hypothesis_id") or ""),
            problem_type=str(data.get("problem_type") or "unknown"),
            allowed_roots=list(data.get("allowed_roots") or []),
            existing_files=list(data.get("existing_files") or []),
            data_inventory=list(data.get("data_inventory") or []),
            profile_summary_json=json.dumps(profile, indent=2, ensure_ascii=False)[:8000],
            baseline_choice_json=json.dumps(choice, indent=2, ensure_ascii=False)[:4000],
        )

    def _run_rule_engine(self, context: StructuredContext) -> CodeProposal:
        """No offline template pack — LLM must generate from inventory."""
        return CodeProposal(
            summary="Awaiting LLM codegen",
            rationale="Jinja scaffolds disabled; generate from profile inventory",
            files=[],
        )
