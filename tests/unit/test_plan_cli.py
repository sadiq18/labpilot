"""CLI tests for ``research plan`` (Plan 5)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.experiments.hypothesis import HypothesisStore

runner = CliRunner()
_HELP_ENV = {"COLUMNS": "200", "NO_COLOR": "1"}


def _plain(text: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def _seed_hypothesis(knowledge: Path, competition: str = "demo") -> str:
    hyp = HypothesisStore(knowledge, competition).create(
        observation="No SpecAugment in the pipeline",
        reason="augmentation often helps",
        prediction="Add SpecAugment to improve CV",
        confidence=0.7,
        expected_impact=0.01,
        tags=["augmentation", "specaugment"],
    )
    return hyp.id


def test_plan_help_documents_subcommands_and_no_execute() -> None:
    result = runner.invoke(app, ["plan", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    for name in ("create", "show", "list"):
        assert name in plain
    assert "--execute" not in result.output


def test_plan_create_help_documents_flags() -> None:
    result = runner.invoke(app, ["plan", "create", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    for flag in ("--hypothesis", "--priority", "--format", "--knowledge-dir"):
        assert flag in plain
    assert "--execute" not in result.output


def test_plan_create_show_list_round_trip(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    hyp_id = _seed_hypothesis(knowledge)

    create = runner.invoke(
        app,
        [
            "plan",
            "create",
            "demo",
            "--hypothesis",
            hyp_id,
            "--priority",
            "2",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert create.exit_code == 0, create.output
    assert "P-001" in create.output
    assert "Created" in create.output

    plans_dir = knowledge / "demo" / "research" / "plans"
    assert (plans_dir / "P-001.json").is_file()
    assert (plans_dir / "P-001.md").is_file()
    assert not (knowledge / "demo" / "research" / "runs").exists()
    assert not (tmp_path / "runs").exists()

    show = runner.invoke(
        app,
        ["plan", "show", "demo", "P-001", "--knowledge-dir", str(knowledge)],
    )
    assert show.exit_code == 0, show.output
    assert "P-001" in show.output
    assert "Task DAG" in show.output

    listed = runner.invoke(
        app,
        ["plan", "list", "demo", "--status", "ready", "--knowledge-dir", str(knowledge)],
    )
    assert listed.exit_code == 0, listed.output
    assert "P-001" in listed.output
    assert hyp_id in listed.output


def test_plan_create_json_is_valid(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    hyp_id = _seed_hypothesis(knowledge)
    result = runner.invoke(
        app,
        [
            "plan",
            "create",
            "demo",
            "--hypothesis",
            hyp_id,
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["id"] == "P-001"
    assert payload["hypothesis_id"] == hyp_id
    assert payload["generated_by"] == "rule_engine"
    assert len(payload["tasks"]) >= 1
    assert payload["priority"] == 0


def test_plan_create_missing_hypothesis_fails(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    result = runner.invoke(
        app,
        [
            "plan",
            "create",
            "demo",
            "--hypothesis",
            "H-999",
            "--knowledge-dir",
            str(knowledge),
        ],
    )
    assert result.exit_code == 1
    assert "Hypothesis not found" in result.output


def test_plan_show_missing_plan_fails(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    _seed_hypothesis(knowledge)
    result = runner.invoke(
        app,
        ["plan", "show", "demo", "P-999", "--knowledge-dir", str(knowledge)],
    )
    assert result.exit_code == 1
    assert "Plan not found" in result.output
