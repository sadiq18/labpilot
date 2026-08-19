"""CLI tests for ``research context retrieve|explain`` (Context Engine)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from helpers.cli import cli_runner

from labpilot.cli.main import app
from labpilot.workspace import scaffold_workspace

runner = cli_runner()
_HELP_ENV = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "GEMINI_API_KEY": "",
    "OPENAI_API_KEY": "",
    "LABPILOT_LLM_MODE": "cloud",
}


def _plain(text: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


def _seed_workspace(tmp_path: Path, slug: str = "ctxcli") -> Path:
    client = scaffold_workspace(tmp_path / "ws", slug)
    from labpilot.research_engine.workspace_facade import Workspace

    ws = Workspace.from_client(client)
    reports = ws.research_paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "note.md").write_text(
        "mixup helps minority classes on audio competitions",
        encoding="utf-8",
    )
    return client.knowledge_dir


def test_context_help_documents_subcommands() -> None:
    result = runner.invoke(app, ["context", "--help"], env=_HELP_ENV)
    assert result.exit_code == 0, result.output
    plain = _plain(result.stdout)
    assert "retrieve" in plain
    assert "explain" in plain


def test_context_retrieve_json_offline_fixture(tmp_path: Path) -> None:
    knowledge = _seed_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "context",
            "retrieve",
            "ctxcli",
            "-q",
            "mixup minority",
            "--format",
            "json",
            "--knowledge-dir",
            str(knowledge),
        ],
        env=_HELP_ENV,
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["request"]["competition"] == "ctxcli"
    assert "items" in data
    assert "bm25_metrics" in data
    blob = json.dumps(data).lower()
    assert "mixup" in blob
    # Must be ContextBundle shape, not RI ResearchContext.
    assert "techniques" not in data or isinstance(data.get("items"), list)
    assert "built_at" in data


def test_context_explain_shows_scores_and_reasons(tmp_path: Path) -> None:
    knowledge = _seed_workspace(tmp_path)
    result = runner.invoke(
        app,
        [
            "context",
            "explain",
            "ctxcli",
            "-q",
            "mixup minority",
            "--knowledge-dir",
            str(knowledge),
        ],
        env=_HELP_ENV,
    )
    assert result.exit_code == 0, result.output
    out = result.stdout.lower()
    assert "context explain" in out or "explain" in out
    assert "score" in out
    assert "reason" in out or "why included" in out
    assert "mixup" in out
