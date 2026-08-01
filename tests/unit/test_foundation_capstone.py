"""Plan 4 — Foundation capstone: analyze → plan → run (dry) via tools."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.research_engine.tools import build_default_tool_registry
from labpilot.research_engine.workspace_facade import Workspace

runner = CliRunner()
_HELP_ENV = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LABPILOT_LLM_MODE": "cloud",
}


def _seed_minimal_analyze(knowledge: Path, competition: str = "demo") -> None:
    import json

    from labpilot.research_engine.intelligence.paths import ResearchPaths

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


def test_foundation_capstone_tools_dry_run_story(tmp_path: Path) -> None:
    """Offline story: seeded analyze → generate_plan → run_plan(dry) via registry."""
    knowledge = tmp_path / "knowledge"
    _seed_minimal_analyze(knowledge)
    ws = Workspace.from_competition(knowledge, "demo", code_root=tmp_path / "ws")
    ws.ensure_roots()
    registry = build_default_tool_registry()

    plan_result = registry.invoke(
        "generate_plan",
        ws,
        baseline=True,
        llm_client=None,
        priority=0,
    )
    assert plan_result.data["plan_id"] == "P-001"
    assert (knowledge / "demo" / "research" / "plans" / "P-001.json").is_file()

    run_result = registry.invoke(
        "run_plan",
        ws,
        plan_id="P-001",
        dry_run=True,
        install_packages=False,
        llm_client=None,
    )
    assert run_result.data["execution_id"] == "E-001"
    assert run_result.data["status"] == "succeeded"


def test_foundation_capstone_cli_plan_then_run_dry(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_minimal_analyze(knowledge)

    created = runner.invoke(
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
    assert created.exit_code == 0, created.output
    assert "P-001" in created.output

    ran = runner.invoke(
        app,
        [
            "run",
            "--plan",
            "P-001",
            "--competition",
            "demo",
            "--knowledge-dir",
            str(knowledge),
            "--dry-run",
            "--no-install-packages",
        ],
        env=_HELP_ENV,
    )
    assert ran.exit_code == 0, ran.output
    assert "E-001" in ran.output
    assert "succeeded" in ran.output
