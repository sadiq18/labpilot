"""`Workspace.for_branch` splits code from shared state (M11 task 7).

A per-branch worktree is only useful if exactly the right paths move with it.
Move too little and every branch writes its pipeline back into the shared
workspace, defeating the isolation. Move too much and each branch looks for a
competition's data inside an empty worktree that is about to be deleted.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import (
    CompetitionWorkspace,
    WorkspacePaths,
    scaffold_workspace,
)


def _legacy(tmp_path: Path) -> Workspace:
    return Workspace(
        competition="titanic",
        knowledge_dir=tmp_path / "knowledge",
        root=tmp_path / "competitions" / "titanic",
        layout="legacy",
    )


def test_client_layout_moves_code_and_pins_shared_state(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)
    worktree = tmp_path / "worktrees" / "b1"

    branch = shared.for_branch(worktree)

    assert branch.root == worktree.resolve()
    assert branch.pipeline_dir == (worktree / "pipeline").resolve()
    assert branch.artifacts_dir == (worktree / "artifacts").resolve()

    assert branch.data_dir == shared.data_dir
    assert branch.raw_data_dir == shared.raw_data_dir
    assert branch.cache_dir == shared.cache_dir
    assert branch.effective_runs_dir == shared.effective_runs_dir


def test_legacy_layout_moves_code_and_pins_shared_state(tmp_path: Path) -> None:
    shared = _legacy(tmp_path)
    worktree = tmp_path / "worktrees" / "b1"

    branch = shared.for_branch(worktree)

    assert branch.pipeline_dir == (worktree / "pipeline").resolve()
    assert branch.artifacts_dir == (worktree / "artifacts").resolve()

    assert branch.data_dir == shared.data_dir
    assert branch.cache_dir == shared.cache_dir
    assert branch.effective_runs_dir == shared.effective_runs_dir


def test_a_branch_never_inherits_the_shared_pipeline(tmp_path: Path) -> None:
    """The isolation claim, stated as the thing that must not happen.

    Resolving code paths off the client marker instead of `root` would leave
    every branch writing into the shared workspace — the worktree would exist
    but change nothing.
    """
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)

    branch = shared.for_branch(tmp_path / "worktrees" / "b1")

    assert branch.pipeline_dir != shared.pipeline_dir
    assert branch.artifacts_dir != shared.artifacts_dir
    assert not branch.pipeline_dir.is_relative_to(shared.root)


def test_custom_layout_names_survive_the_branch(tmp_path: Path) -> None:
    """A workspace that renames its directories keeps those names per branch.

    Falling back to hardcoded "pipeline"/"data" would silently point a
    customised workspace at directories it does not use.
    """
    root = tmp_path / "titanic"
    root.mkdir(parents=True)
    client = CompetitionWorkspace(
        root=root,
        competition="titanic",
        paths=WorkspacePaths(pipeline="src/pipe", data="datasets", cache=".cache2"),
    )
    shared = Workspace.from_client(client)
    worktree = tmp_path / "worktrees" / "b1"

    branch = shared.for_branch(worktree)

    assert branch.pipeline_dir == (worktree / "src" / "pipe").resolve()
    assert branch.data_dir == (root / "datasets").resolve()
    assert branch.cache_dir == (root / ".cache2").resolve()


def test_branching_a_branch_keeps_the_original_shared_dirs(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)

    first = shared.for_branch(tmp_path / "worktrees" / "b1")
    second = first.for_branch(tmp_path / "worktrees" / "b2")

    assert second.data_dir == shared.data_dir
    assert second.cache_dir == shared.cache_dir
    assert second.effective_runs_dir == shared.effective_runs_dir
    assert second.pipeline_dir == (tmp_path / "worktrees" / "b2" / "pipeline").resolve()


def test_unbranched_workspaces_resolve_exactly_as_before(tmp_path: Path) -> None:
    """Regression guard for rewriting the path properties root-relative.

    Without a branch, `root` and the client root are the same directory, so
    every derived path must be byte-identical to what the marker reports.
    """
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    ws = Workspace.from_client(client)

    assert ws.data_dir == client.data_dir
    assert ws.pipeline_dir == client.pipeline_dir
    assert ws.artifacts_dir == client.artifacts_dir
    assert ws.cache_dir == client.cache_dir

    legacy = _legacy(tmp_path)
    assert legacy.data_dir == (legacy.root / "data").resolve()
    assert legacy.pipeline_dir == (legacy.root / "pipeline").resolve()
    assert legacy.artifacts_dir == (legacy.root / "artifacts").resolve()
    assert legacy.cache_dir == (legacy.root / ".cache").resolve()


def test_ensure_roots_creates_the_pinned_dirs_not_branch_copies(tmp_path: Path) -> None:
    """`ensure_roots` on a branch must not materialise a private data dir."""
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)
    worktree = tmp_path / "worktrees" / "b1"

    branch = shared.for_branch(worktree).ensure_roots()

    assert (worktree / "pipeline").is_dir()
    assert not (worktree / "data").exists()
    assert not (worktree / ".cache").exists()
    assert branch.data_dir.is_dir()


def test_ensure_roots_leaves_a_branchs_tracked_gitignore_alone(tmp_path: Path) -> None:
    """`.gitignore` is tracked. Appending to the copy inside a worktree dirties
    the branch before its experiment starts, and the edit lands in the
    snapshot commit and `files_changed` as if the experiment made it.
    """
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)
    worktree = tmp_path / "worktrees" / "b1"
    worktree.mkdir(parents=True)
    # A checkout whose committed .gitignore predates a newly-added pattern.
    (worktree / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

    shared.for_branch(worktree).ensure_roots()

    assert (worktree / ".gitignore").read_text(encoding="utf-8") == "*.pyc\n"


def test_the_shared_workspace_still_reconciles_its_ignores(tmp_path: Path) -> None:
    """The skip is for branches only — the campaign's own workspace is where
    a newly-added pattern has to land."""
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    shared = Workspace.from_client(client)
    (Path(client.root) / ".gitignore").write_text("*.pyc\n", encoding="utf-8")

    shared.ensure_roots()

    assert (Path(client.root) / ".gitignore").read_text(encoding="utf-8") != "*.pyc\n"
