"""Tests for baseline plan compiler + CLI (Research Engineer Plan 3)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner import (
    BaselinePlanError,
    compile_baseline_plan,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

runner = cli_runner()
_HELP_ENV = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    # Force offline plan create (no Gemini/Ollama hang in unit tests).
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LABPILOT_LLM_MODE": "cloud",
}

_BASELINE_TYPES = {
    TaskType.PREPARE_WORKSPACE,
    TaskType.READ_CODE,
    TaskType.WRITE_CODE,
    TaskType.MODIFY_CONFIG,
    TaskType.RESEARCH_REVIEW,
    TaskType.INSTALL_PACKAGE,
    TaskType.RUN_UNIT_TEST,
    TaskType.RUN_SMOKE_TEST,
    TaskType.SELECT_RUNTIME,
    TaskType.RUN_TRAINING,
    TaskType.RUN_INFERENCE,
    TaskType.EVALUATE,
    TaskType.BUILD_SUBMISSION,
    TaskType.GENERATE_REPORT,
    TaskType.REFLECT,
    TaskType.UPDATE_BELIEF,
}


def _plain(text: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def _seed_analyze(knowledge: Path, competition: str = "demo") -> None:
    paths = ResearchPaths(knowledge, competition).ensure()
    paths.report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "competition": competition,
                "techniques": {"items": []},
                "retrieval": {"queries": []},
            }
        ),
        encoding="utf-8",
    )


def test_compile_baseline_plan_offline(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    plan = compile_baseline_plan(
        "demo", knowledge_dir=knowledge, llm_client=None
    )
    assert plan.id == "P-001"
    assert plan.metadata.get("plan_kind") == "baseline"
    assert plan.hypothesis_id == "H-BASELINE"
    assert plan.generated_by == "rule_engine"
    types = {t.type for t in plan.tasks}
    assert _BASELINE_TYPES <= types
    assert (knowledge / "demo" / "research" / "plans" / "P-001.json").is_file()
    assert (knowledge / "demo" / "hypotheses" / "H-BASELINE.json").is_file()


def test_compile_baseline_requires_analyze(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    with pytest.raises(BaselinePlanError, match="Analyze context missing"):
        compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)


def test_compile_baseline_rejects_duplicate(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)
    with pytest.raises(BaselinePlanError, match="already has"):
        compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)


def test_plan_create_baseline_cli(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    create = runner.invoke(
        app,
        [
            "plan",
            "create",
            "demo",
            "--baseline",
            "--knowledge-dir",
            str(knowledge),
        ],
        env=_HELP_ENV,
    )
    assert create.exit_code == 0, create.output
    assert "P-001" in create.output
    assert "baseline" in create.output.lower()
    assert "H-BASELINE" in create.output

    listed = runner.invoke(
        app,
        ["plan", "list", "demo", "--knowledge-dir", str(knowledge)],
        env=_HELP_ENV,
    )
    assert listed.exit_code == 0, listed.output
    assert "P-001" in listed.output
    assert "baseline" in listed.output.lower()


def test_plan_create_baseline_duplicate_cli(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    args = [
        "plan",
        "create",
        "demo",
        "--baseline",
        "--knowledge-dir",
        str(knowledge),
    ]
    assert runner.invoke(app, args, env=_HELP_ENV).exit_code == 0
    second = runner.invoke(app, args, env=_HELP_ENV)
    assert second.exit_code == 1
    assert "refused" in second.output.lower() or "already" in second.output.lower()


def test_compile_baseline_rejects_llm_revision_without_prepare_workspace(
    tmp_path: Path,
) -> None:
    """Gemini previously dropped prepare_workspace; keep rule_engine spine."""
    from labpilot.research_engine.planner.schemas.draft import DraftTask, ResearchPlanDraft
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)

    class FakeLLM:
        def complete(self, system: str, user: str) -> str:
            draft = ResearchPlanDraft(
                goal="broken baseline",
                current_state="x",
                expected_outcome="y",
                tasks=[
                    DraftTask(
                        key="read",
                        type=TaskType.READ_CODE,
                        description="skip workspace",
                    ),
                    DraftTask(
                        key="write",
                        type=TaskType.WRITE_CODE,
                        description="write without profile",
                        depends_on=["read"],
                    ),
                ],
            )
            return draft.model_dump_json()

    plan = compile_baseline_plan(
        "demo", knowledge_dir=knowledge, llm_client=FakeLLM()
    )
    assert plan.generated_by == "rule_engine"
    assert any(t.type == TaskType.PREPARE_WORKSPACE for t in plan.tasks)
    assert any("prepare_workspace" in n for n in plan.notes)


def test_plan_create_help_documents_baseline() -> None:
    result = runner.invoke(app, ["plan", "create", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    assert "--baseline" in plain
    assert "--hypothesis" in plain
