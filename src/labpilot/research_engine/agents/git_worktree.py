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

## The path invariant

Every path this module deletes is **resolved** and **strictly inside**
`.worktrees/`. Both properties are carried by `_SafeTarget`, not by checks at
call sites, and `_force_unregister` — the only function here that deletes
anything — accepts nothing else.

If you add an operation that removes or overwrites anything here, take a
`_SafeTarget`. Normalising an *input* at a public entry point is fine — that
is what `Path(repo_root).resolve()` and `experiment_worktree_root` do. What
must not reappear is a *decision* about whether a path is safe made anywhere
but `_SafeTarget._check`.

Why a type rather than a convention, and the four review rounds that argued
for it, are recorded in
`docs/research-os/autonomy-roadmap/design/05-parallel-branches.md` §8.

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
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from labpilot.research_engine.agents.git_evolution import research_branch_name
from labpilot.research_engine.git import GitTool, open_git_tool
from labpilot.workspace import WORKTREE_DIRNAME

logger = logging.getLogger(__name__)

# Why creation retries at all — this is git's race, not ours.
#
# `git worktree add` registers itself by writing `.git/worktrees/<name>/gitdir`
# and *then* `.git/worktrees/<name>/commondir`, neither atomically. Every git
# command that enumerates worktrees — `add` and `remove` included, because both
# call `get_worktrees()` before doing anything — reads that registration, and a
# registration whose `gitdir` exists while `commondir` is still zero bytes is
# fatal rather than skipped:
#
#     fatal: failed to read .git/worktrees/exp-3/commondir: Undefined error: 0
#
# So K threads calling `add` at once fail each other, at a measured ~3.6% of
# 6-way and 8-way fan-outs on git 2.47.1 (250 rounds each). That was enough to
# flake this module's concurrency test once in a full-suite run and then hide
# for three clean reruns. Under M11 task 7 it would abort a campaign branch.
#
# The window is microseconds wide and closes on its own, so a retry is the
# whole fix. It retries *any* failure rather than matching git's message
# because that message is translated (`fatal: failed to read %s` goes through
# gettext), so a non-English git would sail past a string match — and every
# operation the retry repeats is idempotent by construction anyway:
# `_force_unregister` tolerates absence and `add -B` force-resets. A permanent
# failure therefore costs the backoff budget below and is then re-raised as
# git wrote it.
_WORKTREE_ADD_ATTEMPTS = 4
_WORKTREE_RETRY_BASE_SECONDS = 0.05

__all__ = [
    "WORKTREE_DIRNAME",
    "ExperimentWorktree",
    "ReconcileResult",
    "create_experiment_worktree",
    "experiment_worktree",
    "experiment_worktree_root",
    "is_git_repo",
    "list_registered_worktrees",
    "reconcile_worktrees",
    "remove_experiment_worktree",
]


@dataclass(frozen=True)
class ExperimentWorktree:
    """A checked-out branch with its own directory."""

    path: Path
    branch: str
    repo_root: Path


@dataclass(frozen=True)
class ReconcileResult:
    """What the startup sweep actually managed to clear.

    Both collections are returned rather than only the successes, because the
    caller's decision depends on the failures: a worktree that could not be
    removed still holds its branch checked out, so the experiment keys under
    `failed` will die inside `create_experiment_worktree` later with git's
    "already checked out". A log line cannot be acted on by the campaign
    startup that needs to decide whether to fan out over those keys.

    Tuples, not lists: `frozen=True` stops the fields being rebound but not a
    list being mutated in place, and `ok` is derived from `failed`, so an
    in-place edit would retroactively change the verdict this record exists
    to preserve.
    """

    removed: tuple[Path, ...]
    failed: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        return not self.failed


def experiment_worktree_root(repo_root: Path) -> Path:
    """Where this module's worktrees live. Always resolved, so callers never
    have to normalise it before comparing — that ad-hoc normalising is what
    let `create` and `reconcile` disagree about the same directory."""
    return (Path(repo_root) / WORKTREE_DIRNAME).resolve()


@dataclass(frozen=True)
class _SafeTarget:
    """A path this module is permitted to delete: resolved, and strictly
    inside the experiment worktree root.

    Obtainable only through `under()` / `try_under()`, which resolve and
    verify — so a destructive operation cannot skip either property, because
    it cannot construct the argument. `value` being always resolved also makes
    it the single normal form every path leaving this module is in, so the
    module's own return values compare equal to each other.

    Design rationale and the review history that produced this shape:
    `docs/research-os/autonomy-roadmap/design/05-parallel-branches.md` §8.
    """

    value: Path

    @staticmethod
    def _check(root: Path, path: Path) -> tuple[Path | None, str]:
        """Resolve and verify. Returns `(resolved, "")` or `(None, reason)`.

        The reason is returned rather than re-derived by the caller: the rule
        and its explanation are then one piece of logic, so editing the
        boundary here cannot leave `under()` reporting the wrong cause.

        "Strictly inside" is deliberate: `Path.is_relative_to` is `True` for
        an equal path, but the root holds *every* branch's checkout, so
        deleting it is categorically worse than the escape this was first
        written for — and several branch names reach it (`./.`, `a/..`).
        """
        resolved_root = root.resolve()
        resolved = path.resolve()
        if resolved == resolved_root:
            return None, (
                f"resolves to the experiment worktree root {resolved_root} "
                "itself, which holds every branch"
            )
        if not resolved.is_relative_to(resolved_root):
            return None, (
                f"resolves to {resolved}, outside the experiment worktree "
                f"root {resolved_root}"
            )
        return resolved, ""

    @classmethod
    def under(cls, root: Path, path: Path) -> _SafeTarget:
        """Validating constructor for callers that must fail loudly."""
        resolved, reason = cls._check(root, path)
        if resolved is None:
            raise ValueError(f"refusing to operate on {path}: {reason}")
        return cls(value=resolved)

    @classmethod
    def try_under(cls, root: Path, path: Path) -> _SafeTarget | None:
        """Non-raising form, for the unattended sweep.

        `reconcile_worktrees` must *skip* rather than fail on a path it does
        not own — a developer's own worktree elsewhere in the repo is normal
        and must be left alone.
        """
        resolved, _ = cls._check(root, path)
        return None if resolved is None else cls(value=resolved)


def _worktree_path(repo_root: Path, branch: str) -> _SafeTarget:
    """Map `research/<session>/<experiment>` → `.worktrees/<session>/<experiment>`.

    Returns a `_SafeTarget` rather than a `Path` because
    `research_branch_name` does **not** make its output path-safe: its
    `_SAFE` pattern permits `.` and `/`, so `..` passes through intact. A
    branch built from `session_id=".."` resolves outside `.worktrees/`, and
    the `rmtree` in `_force_unregister` would run on it *before* git ever
    rejects the refname.
    """
    suffix = branch.removeprefix("research/")
    root = experiment_worktree_root(repo_root)
    return _SafeTarget.under(root, root / suffix)


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

    Retries a failing `add` a bounded number of times, because K branches
    creating worktrees at once make each other fail through a race inside git
    itself — see `_WORKTREE_ADD_ATTEMPTS` for the mechanism and the measured
    rate. Path validation stays outside the retry: a branch that resolves out
    of the worktree root is wrong on every attempt, not transiently.
    """
    repo_root = Path(repo_root).resolve()
    tool = git or open_git_tool(repo_root)
    branch = research_branch_name(session_id, experiment_key)
    target = _worktree_path(repo_root, branch)

    _add_with_retry(tool, branch, target)
    return ExperimentWorktree(path=target.value, branch=branch, repo_root=repo_root)


def _add_with_retry(tool: GitTool, branch: str, target: _SafeTarget) -> None:
    """`worktree add`, repeated through git's concurrent-registration window.

    The unregister and the `mkdir` are inside the loop deliberately rather
    than hoisted above it: a concurrent `add` can break *this* call's
    `worktree remove` too (it enumerates worktrees the same way), and that
    failure is swallowed by design in `_force_unregister` — so an attempt can
    fail with "already registered" purely because its own cleanup was the
    casualty. Repeating the cleanup is what makes the next attempt a real
    retry rather than a rerun of the same broken state.
    """
    for attempt in range(1, _WORKTREE_ADD_ATTEMPTS + 1):
        _force_unregister(tool, target)
        target.value.parent.mkdir(parents=True, exist_ok=True)
        try:
            tool.execute("worktree", "add", "-B", branch, str(target.value), "HEAD")
            return
        except Exception as exc:  # noqa: BLE001 — re-raised below once spent
            if attempt == _WORKTREE_ADD_ATTEMPTS:
                logger.warning(
                    "git worktree add for branch %s failed on all %d attempts; "
                    "surfacing git's own error",
                    branch,
                    _WORKTREE_ADD_ATTEMPTS,
                )
                raise
            logger.debug(
                "git worktree add for branch %s failed on attempt %d/%d (%s); "
                "retrying",
                branch,
                attempt,
                _WORKTREE_ADD_ATTEMPTS,
                exc,
            )
            time.sleep(_WORKTREE_RETRY_BASE_SECONDS * 2 ** (attempt - 1))


def remove_experiment_worktree(
    worktree: ExperimentWorktree,
    *,
    git: GitTool | None = None,
) -> None:
    """Unregister and delete one worktree. Safe to call twice.

    Re-validates rather than trusting the handle: `ExperimentWorktree` is a
    plain dataclass a caller can construct directly, so its `path` has not
    necessarily been through `_worktree_path`.
    """
    tool = git or open_git_tool(worktree.repo_root)
    target = _SafeTarget.under(
        experiment_worktree_root(worktree.repo_root), worktree.path
    )
    _force_unregister(tool, target)


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
    # Deliberately not stripping: `splitlines()` already removes the line
    # ending, and a path may legally begin or end with whitespace. Stripping
    # produced a path that does not exist, which `_SafeTarget.try_under` then
    # rejected as uncontained — so reconciliation skipped that worktree
    # forever while reporting nothing to clean.
    for line in out.splitlines():
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
    repo_root = Path(repo_root).resolve()
    if git is None and not is_git_repo(repo_root):
        # `open_git_tool` runs `Repo.init()` plus a bootstrap commit when the
        # directory is not a repository. That is tolerable for the paths that
        # exist to *do* git work, but this one is an unattended startup sweep:
        # creating a repository in a workspace the user deliberately keeps out
        # of git would be a side effect nobody could predict from the name,
        # and there is nothing to reconcile without a repo anyway.
        logger.debug("%s is not a git repository; nothing to reconcile", repo_root)
        return ReconcileResult(removed=(), failed=())
    tool = git or open_git_tool(repo_root)
    own_root = experiment_worktree_root(repo_root)
    removed: list[Path] = []
    failed: list[Path] = []

    for path, branch in list_registered_worktrees(repo_root, git=tool).items():
        target = _SafeTarget.try_under(own_root, path)
        if target is None:
            # Not ours: a developer's own worktree elsewhere in the repo is
            # normal and an unattended sweep must leave it alone. The root
            # itself lands here too — also not ours to delete — but that one
            # is anomalous enough to say out loud, since every branch creation
            # fails while it is registered.
            if path == own_root:
                logger.warning(
                    "%s is itself registered as a git worktree; experiment "
                    "branches cannot be created under it until that is cleared",
                    own_root,
                )
            continue
        if branch is not None and branch in live_branches:
            continue
        _force_unregister(tool, target)
        # Report what actually went, not what was attempted: `_force_unregister`
        # swallows its failures by design (a missing worktree is the normal
        # case), so claiming a removal without checking would let a read-only
        # or busy directory read as reconciled — and the caller would then fan
        # out believing the branch is free.
        if path.exists():
            failed.append(path)
        else:
            removed.append(path)

    # Belt-and-braces for registrations whose directory vanished without a
    # `remove`. Only worth a subprocess when this sweep actually touched
    # something — with nothing under `.worktrees/`, `worktree list` has
    # already told us there is nothing of ours to prune, and this runs on the
    # startup path of every campaign.
    if removed or failed:
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
    return ReconcileResult(removed=tuple(removed), failed=tuple(failed))


def is_git_repo(root: Path) -> bool:
    """Does `root` hold a repository, without creating one to find out?

    Public because every unattended caller needs it: `open_git_tool` runs
    `Repo.init()` plus a bootstrap commit on a directory that is not a
    repository, so anything reaching for git on a startup path has to ask
    first or risk creating one where the user wanted none.

    Matches what `open_git_tool` would accept: GitPython's `Repo(root)` does
    not search parent directories, so a plain subdirectory of a repo is not
    one for our purposes either. `.git` is a file rather than a directory
    inside a linked worktree, so both are accepted.
    """
    return (Path(root) / ".git").exists()


def _force_unregister(tool: GitTool, target: _SafeTarget) -> None:
    """Drop a worktree registration and its directory, tolerating absence.

    Takes a `_SafeTarget`, not a `Path`, and that signature is the whole
    point: this is the only function in the module that deletes anything, so
    requiring the validated type here means no caller — including one added
    later — can reach `rmtree` with a path that has not been resolved and
    proven to sit inside the experiment worktree root. See `_SafeTarget` for
    the four rounds of review that argued for a type over a convention.

    `worktree remove` fails when the path was never registered, which is the
    normal case on first creation — so its failure is expected rather than
    exceptional, and the directory is cleaned separately in case the
    registration and the directory disagree (they do, after a crash).
    """
    if not isinstance(target, _SafeTarget):  # pragma: no cover — guards the future
        # Not redundant with the annotation: this repo runs no static type
        # checker, so the signature is documentation and this is the check.
        # Turning a wrong call into a loud TypeError beats an AttributeError
        # deep inside a delete path, and beats deleting the wrong directory.
        raise TypeError(
            f"_force_unregister requires a _SafeTarget, got {type(target).__name__}; "
            "construct one with _SafeTarget.under()/try_under() so the path is "
            "resolved and proven inside the experiment worktree root"
        )
    path = target.value
    try:
        tool.execute("worktree", "remove", "--force", str(path))
    except Exception:  # noqa: BLE001 — not registered, or already gone
        logger.debug("worktree remove skipped for %s", path, exc_info=True)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
