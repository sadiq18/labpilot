"""Competition workspace discovery, scaffold, and legacy path fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from labpilot.workspace import (
    MARKER_NAME,
    apply_workspace_to_config,
    competition_workspace_path,
    discover_workspace,
    init_git_repo,
    load_config_for_cwd,
    resolve_competition_arg,
    scaffold_workspace,
)


def test_scaffold_and_discover(tmp_path: Path) -> None:
    root = tmp_path / "kaggle"
    ws = scaffold_workspace(root / "demo-comp", "demo-comp", labpilot_hint="/lab")
    assert (ws.root / MARKER_NAME).is_file()
    assert (ws.root / ".gitignore").is_file()
    assert (ws.root / ".env.example").is_file()
    assert "KAGGLE_API_TOKEN" in (ws.root / ".env.example").read_text()
    assert (ws.root / "pipeline").is_dir()
    assert (ws.root / "knowledge").is_dir()
    assert (ws.root / "configs" / "default.yaml").is_file()
    assert "data/" in (ws.root / ".gitignore").read_text()

    found = discover_workspace(start=ws.root / "pipeline")
    assert found is not None
    assert found.competition == "demo-comp"
    assert found.root == ws.root


def test_discover_via_shell_pwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``uv run --directory`` chdirs away; shell PWD should still find the marker."""
    ws = scaffold_workspace(tmp_path / "pwd-comp", "pwd-comp")
    monkeypatch.chdir(tmp_path)  # not the workspace
    monkeypatch.setenv("PWD", str(ws.root))
    found = discover_workspace()
    assert found is not None
    assert found.competition == "pwd-comp"


def test_discover_returns_none_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PWD", raising=False)
    assert discover_workspace() is None


def test_legacy_competition_workspace_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    path = competition_workspace_path(knowledge, "titanic")
    assert path == tmp_path / "competitions" / "titanic"


def test_workspace_mode_competition_workspace_path(tmp_path: Path) -> None:
    ws = scaffold_workspace(tmp_path / "titanic", "titanic")
    path = competition_workspace_path(ws.knowledge_dir, "titanic")
    assert path == ws.root


def test_resolve_competition_defaults_and_mismatch(tmp_path: Path) -> None:
    ws = scaffold_workspace(tmp_path / "slug-a", "slug-a")
    assert resolve_competition_arg(None, ws) == "slug-a"
    assert resolve_competition_arg("slug-a", ws) == "slug-a"
    with pytest.raises(ValueError, match="does not match"):
        resolve_competition_arg("other", ws)
    with pytest.raises(ValueError, match="required"):
        resolve_competition_arg(None, None)


def test_load_config_for_cwd_applies_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = scaffold_workspace(tmp_path / "biohub", "biohub")
    monkeypatch.chdir(ws.root)
    config, found = load_config_for_cwd(config_path=Path("configs/default.yaml"))
    assert found is not None
    assert found.competition == "biohub"
    assert config.knowledge_dir == ws.knowledge_dir
    assert config.kaggle.cache_dir == ws.cache_dir / "kaggle"
    # Explicit override wins
    override = tmp_path / "other-knowledge"
    config2, _ = load_config_for_cwd(
        config_path=Path("configs/default.yaml"),
        knowledge_dir=override,
    )
    assert config2.knowledge_dir == override


def test_apply_workspace_to_config(tmp_path: Path) -> None:
    from labpilot.config import AppConfig

    ws = scaffold_workspace(tmp_path / "x", "x")
    config = AppConfig()
    apply_workspace_to_config(config, ws)
    assert config.knowledge_dir == ws.knowledge_dir


def test_init_git_repo(tmp_path: Path) -> None:
    ws = scaffold_workspace(tmp_path / "g", "g")
    init_git_repo(ws.root)
    assert (ws.root / ".git").is_dir()


def test_scaffold_refuses_nonempty_without_force(tmp_path: Path) -> None:
    target = tmp_path / "busy"
    target.mkdir()
    (target / "existing.txt").write_text("nope")
    with pytest.raises(FileExistsError):
        scaffold_workspace(target, "busy")
    scaffold_workspace(target, "busy", force=True)
    assert (target / MARKER_NAME).is_file()


def test_marker_schema(tmp_path: Path) -> None:
    ws = scaffold_workspace(tmp_path / "m", "m")
    raw = yaml.safe_load((ws.root / MARKER_NAME).read_text())
    assert raw["schema_version"] == 1
    assert raw["competition"] == "m"
    assert raw["paths"]["pipeline"] == "pipeline"
