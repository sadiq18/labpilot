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

from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import (
    LARGE_INPUT_IGNORES,
    REQUIRED_IGNORES,
    SHARED_STATE_IGNORES,
    WORKTREE_DIRNAME,
    ensure_required_ignores,
    scaffold_workspace,
)

#: Everything `ensure_required_ignores` reconciles, across all three groups.
ALL_IGNORES = (*REQUIRED_IGNORES, *SHARED_STATE_IGNORES, *LARGE_INPUT_IGNORES)


def test_appends_missing_patterns_to_an_old_gitignore(tmp_path: Path) -> None:
    """The case the previous fix could not reach."""
    root = tmp_path / "titanic"
    root.mkdir()
    # A workspace scaffolded before these patterns existed, plus a user's own
    # customisation that must survive. `data/` is already present, the way a
    # real old workspace's template-written line would be — it must not be
    # reported as added a second time.
    (root / ".gitignore").write_text(
        "# Competition data (often huge)\ndata/\n\n# my own thing\nscratch/\n",
        encoding="utf-8",
    )

    added = ensure_required_ignores(root)
    assert set(added) == set(ALL_IGNORES) - {"data/"}

    text = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in ALL_IGNORES:
        assert pattern in text
    # User customisation preserved, not rewritten.
    assert "scratch/" in text
    assert "# my own thing" in text
    # The retrofit must not write a second, differently-worded "competition
    # data" section next to the one already there — it has to recognise the
    # pre-existing header as the same group, not invent a third spelling.
    assert text.count("Competition data") == 1


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
    assert set(added) == set(ALL_IGNORES) - {already, "data/"}
    # Not duplicated.
    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert text.count(already) == 1


def test_a_groups_header_is_not_duplicated_across_calls(tmp_path: Path) -> None:
    """A group whose header already ran must not gain a second one.

    Reproduces the case a bare per-pattern check misses: the header is
    present, only one of its group's patterns is. Simulates a workspace
    reconciled before a new pattern joined `SHARED_STATE_IGNORES` — the
    header from that earlier run must not be re-emitted for the new pattern.
    """
    root = tmp_path / "titanic"
    root.mkdir()
    header = "# Bulk research state — never copied into a per-branch worktree"
    already_satisfied = SHARED_STATE_IGNORES[0]
    (root / ".gitignore").write_text(f"{header}\n{already_satisfied}\n", encoding="utf-8")

    added = ensure_required_ignores(root)
    assert set(added) == {"/runs/"} | set(REQUIRED_IGNORES) | set(LARGE_INPUT_IGNORES)

    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert text.count(header) == 1
    assert "/runs/" in text


def test_the_original_groups_header_is_not_duplicated_either(tmp_path: Path) -> None:
    """The header-dedup fix, exercised against the group it was found in.

    The regression test above only ever covered `SHARED_STATE_IGNORES` — the
    group added by this PR. `REQUIRED_IGNORES` is the original group the bug
    was found in; nothing pinned that a fix scoped to the new group's header
    (by name, by hardcoded string, by any means other than the loop's own
    `header` variable) actually generalised to the old one.
    """
    root = tmp_path / "titanic"
    root.mkdir()
    header = "# Machine-local artifacts (locks, temp files, DB sidecars)"
    already_satisfied = REQUIRED_IGNORES[0]
    (root / ".gitignore").write_text(f"{header}\n{already_satisfied}\n", encoding="utf-8")

    added = ensure_required_ignores(root)
    assert set(added) == set(REQUIRED_IGNORES[1:]) | set(SHARED_STATE_IGNORES) | set(
        LARGE_INPUT_IGNORES
    )

    text = (root / ".gitignore").read_text(encoding="utf-8")
    assert text.count(header) == 1


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
    # Simulate a workspace scaffolded before the patterns were added. `data/`
    # is present, `.cache/` and `models/` are not — the retrofit gap large
    # inputs had until `LARGE_INPUT_IGNORES` joined the reconciled groups.
    gitignore.write_text("data/\n.env\n", encoding="utf-8")

    Workspace.from_client(client).ensure_roots()

    text = gitignore.read_text(encoding="utf-8")
    for pattern in ALL_IGNORES:
        assert pattern in text


def test_retrofit_closes_the_large_input_gap_git_worktree_add_would_hit(
    tmp_path: Path,
) -> None:
    """The doc claimed `.cache/`/`models/` were 'already safe'; only true if
    `.gitignore` has the line. Proves the retrofit path actually closes it —
    not just that the pattern gets added, but that git agrees afterward.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    client = scaffold_workspace(root / "titanic", "titanic")
    ws_root = Path(client.root)
    # A `.gitignore` missing the large-input lines, as an old workspace's
    # would be — `data/` present (the original template always had it),
    # `.cache/`/`models/` absent.
    (ws_root / ".gitignore").write_text("data/\n", encoding="utf-8")
    cache_file = ws_root / ".cache" / "kaggle" / "titanic.zip"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text("x", encoding="utf-8")

    before = subprocess.run(
        ["git", "check-ignore", "-q", str(cache_file.relative_to(root))],
        cwd=root,
        check=False,
    )
    assert before.returncode != 0, "fixture didn't reproduce the gap"

    ensure_required_ignores(ws_root)

    after = subprocess.run(
        ["git", "check-ignore", "-q", str(cache_file.relative_to(root))],
        cwd=root,
        check=False,
    )
    assert after.returncode == 0, "retrofit did not close the .cache/ gap"


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
    """The operator needs to know *which* patterns to add by hand.

    `data/` is already present, the way a real old workspace's `.gitignore`
    would be — so the failure path is exercised with a group that is only
    partly missing, not the easy all-or-nothing case.
    """
    root = tmp_path / "titanic"
    root.mkdir()
    (root / ".gitignore").write_text("# Competition data (often huge)\ndata/\n", encoding="utf-8")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", _boom)
    with caplog.at_level(logging.WARNING):
        assert ensure_required_ignores(root) == []

    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "Could not add" in logged
    for pattern in ALL_IGNORES:
        if pattern == "data/":
            continue  # already present — must not be reported as missing
        assert pattern in logged
    assert "data/" not in logged


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
        # Per-experiment worktrees (M11 task 3): a whole checked-out tree per
        # branch, so an unmatched pattern here makes K copies of the working
        # tree committable.
        ws_root / WORKTREE_DIRNAME / "S-001" / "E-001" / "train.py": True,
        # Bulk state (M11): tracked, each of these rode into every worktree —
        # 105 MB per branch on a measured workspace. Both the bare top-level
        # spelling and the real path — resolved the way every store actually
        # resolves it, via `ResearchPaths`, one directory deeper — since a
        # fixture that only ever tested the former stayed green after the
        # pattern regressed to matching nothing real.
        ws_root / "knowledge" / "knowledge.db": True,
        ResearchPaths(ws_root / "knowledge", "titanic").db_path: True,
        ws_root / "runs" / "E-001" / "oof.csv": True,
        # ...but the hypothesis JSONs beside the database are small and stay
        # tracked, which is why the pattern names the file and not `knowledge/`.
        hyp_dir / "H-001.json": False,  # real data — must stay tracked
        # Anchored (leading `/`), not `**/`: a same-named path nested under
        # tracked code is unrelated and must stay tracked.
        ws_root / "pipeline" / "runs" / "some_output.log": False,
        ws_root / "pipeline" / "knowledge.db": False,
        # Large inputs (retrofit path, not just the fresh-scaffold template).
        ws_root / ".cache" / "kaggle" / "titanic.zip": True,
        ws_root / "models" / "checkpoint.pt": True,
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
