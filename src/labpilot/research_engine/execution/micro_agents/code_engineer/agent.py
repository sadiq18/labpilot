"""``CodeEngineerAgent`` — propose full pipeline code as a typed CodeProposal.

Option B: rule_engine returns a Jinja-rendered baseline (full template code).
LLM may revise/replace that into a richer proposal. Never writes disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)

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
        jinja = data.get("jinja_baseline") or {}
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
            problem_type=str(data.get("problem_type") or "tabular_classification"),
            allowed_roots=list(data.get("allowed_roots") or []),
            existing_files=list(data.get("existing_files") or []),
            jinja_baseline_json=json.dumps(jinja, indent=2, ensure_ascii=False)[:12000],
        )

    def _run_rule_engine(self, context: StructuredContext) -> CodeProposal:
        """Return Jinja baseline files if provided; else empty (caller may stub)."""
        jinja = context.data.get("jinja_baseline") or {}
        files = [
            CodeFileSpec(path=path, content=content, action="write")
            for path, content in jinja.items()
            if isinstance(path, str) and isinstance(content, str) and content.strip()
        ]
        if files:
            return CodeProposal(
                summary="Jinja baseline template pack",
                rationale="rule_engine fallback — full template render, not a stub",
                files=files,
            )
        return CodeProposal(
            summary="No Jinja baseline available",
            rationale="rule_engine could not render templates",
            files=[],
        )
