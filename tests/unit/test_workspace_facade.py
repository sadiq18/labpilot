"""Unit tests for Research OS Workspace facade (M1 plan-2b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def test_workspace_from_client_scaffold(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "demo-comp", "demo-comp")
    ws = Workspace.from_client(client, goal="beat baseline")
    assert ws.competition == "demo-comp"
    assert ws.layout == "client"
    assert ws.root == client.root
    assert ws.knowledge_dir == client.knowledge_dir
    assert ws.data_dir == client.data_dir
    assert ws.pipeline_dir == client.pipeline_dir
    assert ws.artifacts_dir == client.artifacts_dir
    assert ws.goal == "beat baseline"
    assert ws.research_paths.competition == "demo-comp"
    assert ws.research_paths.root == client.knowledge_dir / "demo-comp" / "research"


def test_workspace_from_competition_client_layout(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws = Workspace.from_competition(client.knowledge_dir, "titanic")
    assert ws.layout == "client"
    assert ws.root == client.root
    assert ws.artifacts_dir == client.artifacts_dir


def test_workspace_from_competition_legacy_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PWD", raising=False)
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    ws = Workspace.from_competition(knowledge, "titanic")
    assert ws.layout == "legacy"
    assert ws.root == tmp_path / "competitions" / "titanic"
    assert ws.data_dir == ws.root / "data"
    assert ws.pipeline_dir == ws.root / "pipeline"
    assert ws.artifacts_dir == ws.root / "artifacts"
    assert ws.knowledge_dir == knowledge
    ws.ensure_roots()
    assert ws.research_paths.db_path.parent.is_dir()
    assert ws.data_dir.is_dir()


def test_workspace_from_cwd_discovers_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = scaffold_workspace(tmp_path / "biohub", "biohub")
    monkeypatch.chdir(client.root)
    ws = Workspace.from_cwd()
    assert ws.competition == "biohub"
    assert ws.layout == "client"


def test_workspace_from_cwd_legacy_requires_competition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PWD", raising=False)
    with pytest.raises(ValueError, match="competition"):
        Workspace.from_cwd()
    ws = Workspace.from_cwd(competition="demo")
    assert ws.layout == "legacy"
    assert ws.competition == "demo"
    assert ws.knowledge_dir == tmp_path / "knowledge"
