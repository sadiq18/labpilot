"""Plan 3 — CLI strangler: stage commands invoke tools, not engine packages."""

from __future__ import annotations

import ast
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.research_engine.planner import compile_baseline_plan
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

runner = cli_runner()

_CLI_DIR = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "cli"

# Stage CLI modules that must go through ToolRegistry for primary actions.
_STRANGLED = {
    "main.py": {
        # analyze command body must not construct AnalyzeOrchestrator
        "forbidden_names": {"AnalyzeOrchestrator", "write_analysis", "build_default_registry"},
        "required_tokens": ("default_tools", "analyze_competition"),
    },
    "plan.py": {
        "forbidden_names": {"compile_baseline_plan", "compile_research_plan", "PlanArtifacts"},
        "required_tokens": ("default_tools", "generate_plan"),
    },
    "run_engineer.py": {
        # run_plan_command uses tools; resume may still use Engineer
        "required_tokens": ("default_tools", "run_plan"),
    },
    "reflect.py": {
        "forbidden_names": {"run_and_wrap"},
        "required_tokens": ("default_tools", "reflect"),
    },
    "submit.py": {
        "forbidden_names": {"submit_and_learn"},
        "required_tokens": ("default_tools", "submit_learn"),
    },
}


def _seed_analyze(knowledge: Path, competition: str = "demo") -> None:
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


def test_stage_cli_modules_invoke_tools_not_engine_entrypoints() -> None:
    violations: list[str] = []
    for filename, rules in _STRANGLED.items():
        path = _CLI_DIR / filename
        source = path.read_text(encoding="utf-8")
        for token in rules.get("required_tokens", ()):
            if token not in source:
                violations.append(f"{filename}: missing {token!r}")
        tree = ast.parse(source, filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[-1])
        for forbidden in rules.get("forbidden_names", ()):
            if forbidden in names:
                violations.append(f"{filename}: still references {forbidden}")
    assert not violations, "CLI strangler violations:\n" + "\n".join(violations)


def test_cli_strangler_plan_create_via_tool(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    hyp = HypothesisStore(knowledge, "demo").create(
        observation="x",
        reason="y",
        prediction="z",
        confidence=0.5,
        expected_impact=0.01,
    )
    result = runner.invoke(
        app,
        [
            "plan",
            "create",
            "demo",
            "--hypothesis",
            hyp.id,
            "--knowledge-dir",
            str(knowledge),
        ],
        env={
            "GEMINI_API_KEY": "",
            "OPENAI_API_KEY": "",
            "LABPILOT_LLM_MODE": "cloud",
            "NO_COLOR": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "P-001" in result.output
    assert (knowledge / "demo" / "research" / "plans" / "P-001.json").is_file()


def test_cli_strangler_run_dry_via_tool(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_analyze(knowledge)
    compile_baseline_plan("demo", knowledge_dir=knowledge, llm_client=None)
    result = runner.invoke(
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
    )
    assert result.exit_code == 0, result.output
    assert "E-001" in result.output
