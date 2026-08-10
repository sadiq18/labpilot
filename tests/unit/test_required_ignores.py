"""M11: machine-local ignore patterns must reach *existing* workspaces.

`scaffold_workspace` writes `.gitignore` only when it is absent, and it is
reached only by `research init`, which refuses an existing workspace. So a
pattern added to the template alone never arrives anywhere that has already
run a campaign — which is exactly where the lock and temp files accumulate.
`Workspace.ensure_roots()` reconciles them instead.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import (
    REQUIRED_IGNORES,
    ensure_required_ignores,
    scaffold_workspace,
)


def test_appends_missing_patterns_to_an_old_gitignore(tmp_path: Path) -> None:
    """The case the previous fix could not reach."""
    root = tmp_path / "titanic"
    root.mkdir()
    # A workspace scaffolded before these patterns existed, plus a user's own
    # customisation that must survive.
    (root / ".gitignore").write_text(
        "# Competition data (often huge)\ndata/\n\n# my own thing\nscratch/\n",
        encoding="utf-8",
    )

    added = ensure_required_ignores(root)
    assert set(added) == set(REQUIRED_IGNORES)

    text = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in REQUIRED_IGNORES:
        assert pattern in text
    # User customisation preserved, not rewritten.
    assert "scratch/" in text
    assert "# my own thing" in text


def test_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "titanic"
    root.mkdir()
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")

    first = ensure_required_ignores(root)
    assert first
    before = (root / ".gitignore").read_text(encoding="utf-8")

    second = ensure_required_ignores(root)
    assert second == []
    assert (root / ".gitignore").read_text(encoding="utf-8") == before


def test_appends_only_the_genuinely_missing_pattern(tmp_path: Path) -> None:
    root = tmp_path / "titanic"
    root.mkdir()
    already = "**/knowledge.db-wal"
    (root / ".gitignore").write_text(f"data/\n{already}\n", encoding="utf-8")

    added = ensure_required_ignores(root)
    assert already not in added
    assert set(added) == set(REQUIRED_IGNORES) - {already}
    # Not duplicated.
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert text.count(already) == 1


def test_no_gitignore_is_left_alone(tmp_path: Path) -> None:
    """Inventing one would fight a user who deliberately removed it."""
    root = tmp_path / "titanic"
    root.mkdir()
    assert ensure_required_ignores(root) == []
    assert not (root / ".gitignore").is_file()


def test_ensure_roots_reconciles_an_existing_workspace(tmp_path: Path) -> None:
    """End to end: the path `research conduct` actually takes."""
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    gitignore = Path(client.root) / ".gitignore"
    # Simulate a workspace scaffolded before the patterns were added.
    gitignore.write_text("data/\n.env\n", encoding="utf-8")

    Workspace.from_client(client).ensure_roots()

    text = gitignore.read_text(encoding="utf-8")
    for pattern in REQUIRED_IGNORES:
        assert pattern in text


def test_unreadable_gitignore_warns_instead_of_failing_silently(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`[]` also means 'already complete', so a silent failure reads as success."""
    root = tmp_path / "titanic"
    root.mkdir()
    gitignore = root / ".gitignore"
    gitignore.write_text("data/\n", encoding="utf-8")
    gitignore.chmod(0o000)
    try:
        with caplog.at_level(logging.WARNING):
            assert ensure_required_ignores(root) == []
        assert any("Could not read" in r.getMessage() for r in caplog.records)
    finally:
        gitignore.chmod(0o644)


def test_unwritable_gitignore_warns_and_names_the_patterns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator needs to know *which* patterns to add by hand."""
    root = tmp_path / "titanic"
    root.mkdir()
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _boom)
    with caplog.at_level(logging.WARNING):
        assert ensure_required_ignores(root) == []

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "Could not add" in logged
    for pattern in REQUIRED_IGNORES:
        assert pattern in logged


def test_patterns_actually_ignore_the_real_artifact_names(tmp_path: Path) -> None:
    """Patterns are only useful if git agrees they match; assert, don't assume."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    client = scaffold_workspace(root / "titanic", "titanic")
    ws_root = Path(client.root)

    hyp_dir = ws_root / "knowledge" / "titanic" / "hypotheses"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        ws_root / "knowledge" / "knowledge.db.writelock": True,
        hyp_dir / ".H-001.lock": True,
        hyp_dir / ".alloc.lock": True,
        hyp_dir / ".H-001.json.tmp-123-456": True,
        ws_root / "knowledge" / "titanic" / ".knowledge_base.json.lock": True,
        hyp_dir / "H-001.json": False,  # real data — must stay tracked
    }
    for path, _ in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    for path, should_be_ignored in artifacts.items():
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path.relative_to(root))],
            cwd=root,
            check=False,
        )
        ignored = result.returncode == 0
        assert ignored is should_be_ignored, f"{path.name}: ignored={ignored}"
