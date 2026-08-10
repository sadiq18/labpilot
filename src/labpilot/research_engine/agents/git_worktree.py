"""Per-branch git worktrees for parallel experiments (M11 task 3).

`snapshot_before_experiment` creates a branch and checks it out **in place**,
mutating the one working tree every caller shares. That is correct while the
campaign runs one experiment at a time and fatal the moment it does not: K
branches racing `create_branch(checkout=True)` against the same root overwrite
each other's files, so each experiment would train against whatever code the
last checkout happened to leave behind.

A git worktree gives each branch its own directory backed by the same object
store, which is the isolation the fan-out needs without cloning the repo per
branch.

Lifecycle here is deliberately three-layered, because two of them are not
enough:

- `experiment_worktree()` — the context manager. Removes on the way out
  whether the body returned or raised, which covers a branch that fails
  mid-run.
- `remove_experiment_worktree()` — the explicit call, for callers that cannot
  use a `with` block.
- `reconcile_worktrees()` — the startup sweep. Neither of the above survives
  `SIGKILL`, an OOM kill, or a host restart, and a worktree left registered
  keeps its branch checked out so a later run cannot reuse it. Reconciliation
  is what makes the crash case recoverable rather than requiring a human with
  `git worktree prune`.

**Nothing in production calls this yet, and that is the intended state.** It
is the mechanism M11 task 7 consumes: the conductor creates one worktree per
branch before fanning out, runs each `ParallelWorkItem` against its own
`ExperimentWorktree.path` instead of the shared `workspace.root`, and calls
`reconcile_worktrees()` at campaign start with the branches its live steps
own. Until that lands, `snapshot_before_experiment` keeps checking out in
place, which is correct for the sequential (K=1) path it still serves.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from labpilot.research_engine.agents.git_evolution import research_branch_name
from labpilot.research_engine.git import GitTool, open_git_tool

logger = logging.getLogger(__name__)

#: Worktrees live under the workspace root so they are discoverable and share
#: its filesystem (git requires the same volume for a linked worktree). The
#: directory is machine-local and covered by `REQUIRED_IGNORES` in
#: `labpilot.workspace`, so it is never committed and is reconciled into
#: workspaces that predate this feature.
WORKTREE_DIRNAME = ".worktrees"


@dataclass(frozen=True)
class ExperimentWorktree:
    """A checked-out branch with its own directory."""

    path: Path
    branch: str
    repo_root: Path


@dataclass(frozen=True)
class ReconcileResult:
    """What the startup sweep actually managed to clear.

    Both lists are returned rather than only the successes, because the
    caller's decision depends on the failures: a worktree that could not be
    removed still holds its branch checked out, so the experiment keys under
    `failed` will die inside `create_experiment_worktree` later with git's
    "already checked out". A log line cannot be acted on by the campaign
    startup that needs to decide whether to fan out over those keys.
    """

    removed: list[Path]
    failed: list[Path]

    @property
    def ok(self) -> bool:
        return not self.failed


def experiment_worktree_root(repo_root: Path) -> Path:
    return Path(repo_root) / WORKTREE_DIRNAME


def _worktree_path(repo_root: Path, branch: str) -> Path:
    """Map `research/<session>/<experiment>` → `.worktrees/<session>/<experiment>`.

    Containment-checked, because `research_branch_name` does **not** make its
    output path-safe: its `_SAFE` pattern permits `.` and `/`, so `..` passes
    through intact. Without this check a branch built from `session_id=".."`
    resolves outside `.worktrees/`, and `_force_unregister`'s `rmtree` runs on
    that path *before* git ever rejects the refname — deleting, say, the whole
    `knowledge/` directory and then reporting the failure git raised.
    """
    suffix = branch.removeprefix("research/")
    root = experiment_worktree_root(repo_root)
    path = root / suffix
    _assert_contained(path, root)
    return path


def _is_contained(path: Path, root: Path) -> bool:
    """Is `path` strictly inside `root`, resolved against symlinks?

    **The single containment rule for this module.** Every destructive
    operation decides through this one predicate rather than its own
    comparison: three call sites agreeing by inspection is what produced the
    same boundary bug three times running, once in each hand-rolled variant.

    "Strictly inside" is deliberate — `Path.is_relative_to` is `True` for an
    equal path, but the root holds *every* branch's checkout, so deleting it
    is categorically worse than the escape this guard was written for, and
    several branch names reach it (`./.`, `a/..`).
    """
    resolved_root = root.resolve()
    resolved = path.resolve()
    return resolved != resolved_root and resolved.is_relative_to(resolved_root)


def _assert_contained(path: Path, root: Path) -> None:
    """Raising form of `_is_contained`, for callers that cannot skip.

    Only formats the explanation; the rule itself lives in the predicate.
    """
    if _is_contained(path, root):
        return
    resolved_root = root.resolve()
    if path.resolve() == resolved_root:
        raise ValueError(
            f"refusing to operate on {path}: resolves to the experiment "
            f"worktree root {resolved_root} itself, which holds every branch"
        )
    raise ValueError(
        f"refusing to operate on {path}: resolves to {path.resolve()}, outside "
        f"the experiment worktree root {resolved_root}"
    )


def create_experiment_worktree(
    repo_root: Path,
    *,
    session_id: str,
    experiment_key: str,
    git: GitTool | None = None,
) -> ExperimentWorktree:
    """Create (or re-create) an isolated worktree for one experiment branch.

    Uses `worktree add -B`, matching `create_branch(force=True)`'s existing
    force-reset semantics, so re-running an experiment key is not an error.
    Any worktree already registered at the target path is removed first —
    a stale registration from a previous crash would otherwise make `add`
    fail and take the branch down with it.
    """
    repo_root = Path(repo_root)
    tool = git or open_git_tool(repo_root)
    branch = research_branch_name(session_id, experiment_key)
    path = _worktree_path(repo_root, branch)

    _force_unregister(tool, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tool.execute("worktree", "add", "-B", branch, str(path), "HEAD")
    return ExperimentWorktree(path=path, branch=branch, repo_root=repo_root)


def remove_experiment_worktree(
    worktree: ExperimentWorktree,
    *,
    git: GitTool | None = None,
) -> None:
    """Unregister and delete one worktree. Safe to call twice.

    Re-checks containment rather than trusting the handle: an
    `ExperimentWorktree` is a plain dataclass a caller can construct directly,
    so its `path` has not necessarily been through `_worktree_path`.
    """
    tool = git or open_git_tool(worktree.repo_root)
    _assert_contained(worktree.path, experiment_worktree_root(worktree.repo_root))
    _force_unregister(tool, worktree.path)


@contextmanager
def experiment_worktree(
    repo_root: Path,
    *,
    session_id: str,
    experiment_key: str,
    git: GitTool | None = None,
) -> Iterator[ExperimentWorktree]:
    """Own a worktree for the duration of the block, removing it either way.

    The `finally` is the point: a branch that raises mid-experiment must not
    leave its worktree registered, or its branch stays checked out and the
    next run of that experiment key cannot claim it.
    """
    worktree = create_experiment_worktree(
        repo_root,
        session_id=session_id,
        experiment_key=experiment_key,
        git=git,
    )
    try:
        yield worktree
    finally:
        try:
            remove_experiment_worktree(worktree, git=git)
        except Exception:  # noqa: BLE001 — teardown must not mask the body's error
            logger.exception(
                "could not remove experiment worktree %s (branch %s); "
                "startup reconciliation will retry",
                worktree.path,
                worktree.branch,
            )


def list_registered_worktrees(
    repo_root: Path,
    *,
    git: GitTool | None = None,
) -> dict[Path, str | None]:
    """Map registered worktree path → branch name (None when detached).

    Parsed from `worktree list --porcelain`: blank-line-separated records,
    each beginning with a `worktree <path>` line.
    """
    tool = git or open_git_tool(repo_root)
    out = tool.execute("worktree", "list", "--porcelain")
    found: dict[Path, str | None] = {}
    current: Path | None = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :]).resolve()
            found[current] = None
        elif line.startswith("branch ") and current is not None:
            found[current] = line[len("branch ") :].removeprefix("refs/heads/")
    return found


def reconcile_worktrees(
    repo_root: Path,
    *,
    live_branches: set[str],
    git: GitTool | None = None,
) -> ReconcileResult:
    """Remove experiment worktrees no live campaign step still owns.

    Returns what was removed *and* what could not be — see `ReconcileResult`.
    `live_branches` is supplied by the caller rather than queried here so this
    module stays free of any conductor dependency — the caller knows which
    steps are running.

    Only worktrees under `.worktrees/` are considered. A developer's own
    worktree elsewhere in the repo is never touched, which matters because
    this runs unattended at startup.
    """
    repo_root = Path(repo_root)
    tool = git or open_git_tool(repo_root)
    own_root = experiment_worktree_root(repo_root).resolve()
    removed: list[Path] = []
    failed: list[Path] = []

    for path, branch in list_registered_worktrees(repo_root, git=tool).items():
        if not _is_contained(path, own_root):
            # Skipping rather than raising is the point of the predicate form:
            # a developer's own worktree elsewhere in the repo is normal and
            # must be left alone by an unattended sweep. The root itself lands
            # here too — it is not ours to delete — but that one is anomalous
            # enough to say out loud, since every branch creation will fail
            # while it is registered.
            if path == own_root:
                logger.warning(
                    "%s is itself registered as a git worktree; experiment "
                    "branches cannot be created under it until that is cleared",
                    own_root,
                )
            continue
        if branch is not None and branch in live_branches:
            continue
        _force_unregister(tool, path)
        # Report what actually went, not what was attempted: `_force_unregister`
        # swallows its failures by design (a missing worktree is the normal
        # case), so claiming a removal without checking would let a read-only
        # or busy directory read as reconciled — and the caller would then fan
        # out believing the branch is free.
        if path.exists():
            failed.append(path)
        else:
            removed.append(path)

    # Also clears registrations whose directory vanished without `remove`.
    tool.execute("worktree", "prune")
    if removed:
        logger.info("reconciled %d orphaned experiment worktree(s)", len(removed))
    if failed:
        logger.warning(
            "%d experiment worktree(s) could not be removed and still hold their "
            "branches checked out; a later run of the same experiment key will "
            "fail until they are cleared by hand: %s",
            len(failed),
            ", ".join(str(p) for p in failed),
        )
    return ReconcileResult(removed=removed, failed=failed)


def _force_unregister(tool: GitTool, path: Path) -> None:
    """Drop a worktree registration and its directory, tolerating absence.

    `worktree remove` fails when the path was never registered, which is the
    normal case on first creation — so its failure is expected rather than
    exceptional, and the directory is cleaned separately in case the
    registration and the directory disagree (they do, after a crash).
    """
    try:
        tool.execute("worktree", "remove", "--force", str(path))
    except Exception:  # noqa: BLE001 — not registered, or already gone
        logger.debug("worktree remove skipped for %s", path, exc_info=True)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
