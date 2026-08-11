"""M11 task 3: per-branch worktree isolation and crash-safe teardown.

Drives real `git worktree` against a real repo rather than a fake GitTool —
the whole point of the change is that git's own working-tree semantics stop
two branches colliding, and a fake would assert the mock, not the isolation.
"""

from __future__ import annotations

import subprocess
import threading
import traceback
from pathlib import Path

import pytest

from labpilot.research_engine.agents.git_worktree import (
    WORKTREE_DIRNAME,
    ExperimentWorktree,
    create_experiment_worktree,
    experiment_worktree,
    experiment_worktree_root,
    list_registered_worktrees,
    reconcile_worktrees,
    remove_experiment_worktree,
)
from labpilot.research_engine.git import open_git_tool


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    run = lambda *a: subprocess.run(a, cwd=root, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    (root / "train.py").write_text("print('base')\n", encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    return root


def test_every_path_leaving_the_module_is_already_resolved(tmp_path: Path) -> None:
    """One normal form out of this module, so its own values compare equal.

    `create` used to return the path it constructed while `reconcile` and
    `list_registered_worktrees` returned resolved ones, so on any symlinked
    workspace — macOS `/tmp` → `/private/tmp`, the default — a caller asking
    `wt.path in result.removed` got False for a worktree that had just been
    removed. The earlier tests hid it by calling `.resolve()` at each
    comparison; asserting the invariant directly is what stops it returning.
    """
    root = _repo(tmp_path)
    wt = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")

    assert wt.path == wt.path.resolve()
    assert wt.repo_root == wt.repo_root.resolve()
    assert all(p == p.resolve() for p in list_registered_worktrees(root))

    result = reconcile_worktrees(root, live_branches=set())
    assert all(p == p.resolve() for p in result.removed)
    # The correlation task 7 will actually perform, without normalising first.
    assert wt.path in result.removed


def test_module_values_compare_across_a_symlinked_workspace(tmp_path: Path) -> None:
    """The same correlation, reached through a symlink rather than trusting
    that the platform happens to give us one."""
    (tmp_path / "real").mkdir()
    real = _repo(tmp_path / "real")
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "real", target_is_directory=True)

    wt = create_experiment_worktree(link / "ws", session_id="s1", experiment_key="exp-a")
    result = reconcile_worktrees(link / "ws", live_branches=set())

    assert wt.path in result.removed
    assert real.exists()


def test_reconcile_does_not_create_a_repo_in_a_non_git_workspace(tmp_path: Path) -> None:
    """A cleanup sweep must not have `git init` as a side effect.

    `open_git_tool` falls back to `Repo.init()` plus a bootstrap commit for a
    directory that is not a repository. Tolerable where the caller exists to
    do git work; not for an unattended startup sweep against a workspace the
    user deliberately keeps out of git — `init_git_repo` is optional at
    scaffold time.
    """
    plain = tmp_path / "no-git-here"
    plain.mkdir()

    result = reconcile_worktrees(plain, live_branches=set())

    assert result.removed == ()
    assert result.ok
    assert not (plain / ".git").exists(), "reconciliation created a git repository"


def test_reconcile_result_cannot_be_mutated_after_the_fact(tmp_path: Path) -> None:
    """`ok` is derived from `failed`, so an in-place edit would rewrite the verdict."""
    root = _repo(tmp_path)
    create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    result = reconcile_worktrees(root, live_branches=set())

    assert isinstance(result.removed, tuple)
    assert isinstance(result.failed, tuple)
    with pytest.raises(AttributeError):
        result.removed.append(Path("/tmp/injected"))  # type: ignore[attr-defined]


def test_worktree_paths_with_trailing_whitespace_are_parsed_intact(
    tmp_path: Path,
) -> None:
    """`splitlines()` already drops the line ending; stripping corrupts paths.

    A stripped path does not exist, so `try_under` rejected it as uncontained
    and reconciliation skipped that worktree forever while reporting nothing
    to clean.
    """
    root = _repo(tmp_path)
    # The trailing space must be on the *last* component: `line.strip()` only
    # reaches the end of the line, so a space mid-path would not exercise it.
    odd = tmp_path / "wt "  # legal directory name on macOS and Linux
    subprocess.run(
        ["git", "worktree", "add", "-b", "odd/space", str(odd), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    seen = list_registered_worktrees(root)

    assert odd.resolve() in seen, f"trailing space lost; parsed {[str(p) for p in seen]}"
    assert all(p.exists() for p in seen), "a parsed path does not exist on disk"


def test_worktree_dirname_is_the_one_the_gitignore_pattern_uses(tmp_path: Path) -> None:
    """One definition of the directory name, not three.

    The name lives in `labpilot.workspace` and `REQUIRED_IGNORES` is built
    from it, so a rename cannot leave the ignore pattern matching a stale
    literal while its own test still passes.
    """
    from labpilot.workspace import REQUIRED_IGNORES

    assert f"{WORKTREE_DIRNAME}/" in REQUIRED_IGNORES
    root = _repo(tmp_path)
    wt = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    assert wt.path.is_relative_to(root / WORKTREE_DIRNAME)


def test_worktree_isolates_edits_between_branches(tmp_path: Path) -> None:
    """The bug this exists to fix: two branches editing the same file."""
    root = _repo(tmp_path)
    a = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    b = create_experiment_worktree(root, session_id="s1", experiment_key="exp-b")

    (a.path / "train.py").write_text("print('A')\n", encoding="utf-8")
    (b.path / "train.py").write_text("print('B')\n", encoding="utf-8")

    # Each branch sees only its own edit, and the shared root is untouched.
    assert (a.path / "train.py").read_text() == "print('A')\n"
    assert (b.path / "train.py").read_text() == "print('B')\n"
    assert (root / "train.py").read_text() == "print('base')\n"
    assert a.branch != b.branch

    remove_experiment_worktree(a)
    remove_experiment_worktree(b)


def test_concurrent_creation_gives_each_thread_its_own_tree(tmp_path: Path) -> None:
    """K branches created in parallel, as the fan-out will do.

    Each worker captures its own exception instead of letting it die inside
    `Thread.run`. Without that, a failure surfaced only as `len(made) != n` —
    a count, naming neither the thread nor git's error — and this test did
    exactly that once in a full-suite run, then passed three reruns with
    nothing to go on. The traceback in the assertion message is the point:
    the next occurrence has to arrive already diagnosed.
    """
    root = _repo(tmp_path)
    n = 6
    barrier = threading.Barrier(n)
    made: list[tuple[str, Path]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def branch(i: int) -> None:
        try:
            barrier.wait()
            wt = create_experiment_worktree(
                root, session_id="s1", experiment_key=f"exp-{i}"
            )
            (wt.path / "train.py").write_text(f"print({i})\n", encoding="utf-8")
            with lock:
                made.append((wt.branch, wt.path))
        except Exception:  # noqa: BLE001 — reporting the cause IS the job here
            with lock:
                errors.append(f"--- thread {i} ---\n{traceback.format_exc()}")

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Assert the cause before the symptom, so a failure reads as git's error
    # rather than as arithmetic.
    assert not errors, "concurrent worktree creation failed:\n" + "\n".join(errors)
    assert len(made) == n
    assert len({b for b, _ in made}) == n
    # Every branch kept its own content — no clobbering.
    for i, (_, path) in enumerate(sorted(made, key=lambda m: m[0])):
        assert (path / "train.py").read_text() == f"print({i})\n"


class _FlakyAdd:
    """A GitTool that fails `worktree add` a fixed number of times first.

    Reproduces git's concurrent-registration window deterministically. The
    real thing needs two threads inside a microsecond-wide race, which is why
    the bug this covers reached main: the concurrency test hit it roughly once
    in thirty runs and could not be made to do it again on demand.
    """

    # git's actual message for the window, verbatim — including that errno is
    # 0, because the read hits EOF on a zero-byte file rather than erroring.
    TRANSIENT = (
        "Preparing worktree (new branch 'research/s1/exp-a')\n"
        "fatal: failed to read .git/worktrees/exp-other/commondir: Undefined error: 0"
    )

    def __init__(self, inner: object, *, failures: int) -> None:
        self._inner = inner
        self._remaining = failures
        self.add_attempts = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def execute(self, *git_args: str) -> str:
        if git_args[:2] == ("worktree", "add"):
            self.add_attempts += 1
            if self._remaining > 0:
                self._remaining -= 1
                raise RuntimeError(self.TRANSIENT)
        return self._inner.execute(*git_args)  # type: ignore[attr-defined]


def test_create_retries_gits_concurrent_registration_window(tmp_path: Path) -> None:
    """A transient `add` failure must not abort the branch.

    `git worktree add` writes `.git/worktrees/<name>/gitdir` before
    `commondir`, and any concurrent git command that enumerates worktrees
    treats that half-written pair as fatal instead of skipping it. K-way
    fan-out therefore makes branches fail each other at a few percent per
    round — survivable only because the window closes on its own.
    """
    root = _repo(tmp_path)
    flaky = _FlakyAdd(open_git_tool(root), failures=2)

    wt = create_experiment_worktree(
        root, session_id="s1", experiment_key="exp-a", git=flaky
    )

    assert wt.path.is_dir()
    assert flaky.add_attempts == 3, "did not retry through the transient window"
    remove_experiment_worktree(wt, git=flaky)


def test_create_surfaces_gits_own_error_once_retries_are_spent(tmp_path: Path) -> None:
    """Bounded, and the error that escapes is git's — not a retry wrapper.

    Retrying blind would turn a permanent failure ("already checked out
    elsewhere") into a silent stall; the budget is what keeps it a failure,
    and re-raising unwrapped is what keeps it diagnosable.
    """
    import labpilot.research_engine.agents.git_worktree as gw

    root = _repo(tmp_path)
    flaky = _FlakyAdd(open_git_tool(root), failures=999)

    with pytest.raises(RuntimeError, match="commondir"):
        create_experiment_worktree(
            root, session_id="s1", experiment_key="exp-a", git=flaky
        )

    assert flaky.add_attempts == gw._WORKTREE_ADD_ATTEMPTS


def test_a_traversing_id_is_refused_without_being_retried(tmp_path: Path) -> None:
    """The retry covers git's window, not the containment guard.

    A branch resolving outside `.worktrees/` is wrong on every attempt, so
    retrying it would only delay the refusal — and would re-run the deleting
    path three more times against a target already judged unsafe.
    """
    root = _repo(tmp_path)
    flaky = _FlakyAdd(open_git_tool(root), failures=0)

    with pytest.raises(ValueError, match="outside the experiment worktree root"):
        create_experiment_worktree(
            root, session_id="..", experiment_key="knowledge", git=flaky
        )

    assert flaky.add_attempts == 0, "validation ran inside the retry loop"


def test_context_manager_removes_on_success(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with experiment_worktree(root, session_id="s1", experiment_key="exp-a") as wt:
        assert wt.path.is_dir()
        held = wt.path
    assert not held.exists()
    assert held not in list_registered_worktrees(root)


def test_context_manager_removes_on_failure(tmp_path: Path) -> None:
    """Mid-run failure must not leave the branch checked out."""
    root = _repo(tmp_path)
    held: Path | None = None
    with pytest.raises(RuntimeError, match="branch blew up"):
        with experiment_worktree(root, session_id="s1", experiment_key="exp-a") as wt:
            held = wt.path
            raise RuntimeError("branch blew up")

    assert held is not None
    assert not held.exists()
    assert held not in list_registered_worktrees(root)


def test_failed_branch_key_can_be_reused_afterwards(tmp_path: Path) -> None:
    """Creation is self-healing, independently of teardown having run.

    Deliberately *not* a teardown test — it passes with the `finally` removed,
    because `create` force-unregisters first. That is the belt-and-braces
    worth pinning: even if teardown and reconciliation both failed, retrying
    the same experiment key still works rather than dying on "already checked
    out". The teardown itself is covered by the two context-manager tests.
    """
    root = _repo(tmp_path)
    with pytest.raises(RuntimeError):
        with experiment_worktree(root, session_id="s1", experiment_key="exp-a"):
            raise RuntimeError("boom")
    # Same key again must succeed, not fail with "already checked out".
    again = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    assert again.path.is_dir()
    remove_experiment_worktree(again)


def test_reconcile_removes_orphans_but_keeps_live_branches(tmp_path: Path) -> None:
    """The crash path: teardown never ran, so startup must clean up."""
    root = _repo(tmp_path)
    live = create_experiment_worktree(root, session_id="s1", experiment_key="live")
    orphan = create_experiment_worktree(root, session_id="s1", experiment_key="orphan")

    result = reconcile_worktrees(root, live_branches={live.branch})

    assert orphan.path in result.removed
    assert live.path not in result.removed
    assert result.ok
    assert not orphan.path.exists()
    assert live.path.is_dir()
    remove_experiment_worktree(live)


def test_reconcile_ignores_worktrees_outside_our_directory(tmp_path: Path) -> None:
    """Runs unattended at startup — must not touch a developer's own worktree."""
    root = _repo(tmp_path)
    outside = tmp_path / "my-own-worktree"
    subprocess.run(
        ["git", "worktree", "add", "-b", "my/work", str(outside), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    result = reconcile_worktrees(root, live_branches=set())

    assert result.removed == ()
    assert outside.is_dir()
    # Still registered under its own branch — reconciliation left it entirely alone.
    assert list_registered_worktrees(root)[outside.resolve()] == "my/work"


def test_reconcile_never_removes_the_worktree_root_itself(tmp_path: Path) -> None:
    """Reconcile's boundary — the case its own filter used to let through.

    `git worktree add .worktrees` is something git permits, and
    `is_relative_to` is True for an equal path, so the root passed the filter
    and was handed to `_force_unregister`. An unattended startup sweep would
    then delete every branch's checkout. Create and remove were covered at
    this boundary; reconcile's filter was not, which is exactly where the
    third instance of this bug lived.
    """
    root = _repo(tmp_path)
    worktree_root = experiment_worktree_root(root)
    subprocess.run(
        ["git", "worktree", "add", "-B", "stray", str(worktree_root), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    result = reconcile_worktrees(root, live_branches=set())

    assert worktree_root not in result.removed
    assert worktree_root.is_dir(), "the root holding every branch was deleted"


def test_reconcile_prunes_a_directory_deleted_out_from_under_git(tmp_path: Path) -> None:
    """Registration and directory disagree after a hard kill."""
    import shutil

    root = _repo(tmp_path)
    wt = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    shutil.rmtree(wt.path)  # directory gone, registration remains

    reconcile_worktrees(root, live_branches=set())

    assert wt.path not in list_registered_worktrees(root)


@pytest.mark.parametrize(
    ("session_id", "experiment_key"),
    [
        ("..", "knowledge"),  # deleted <workspace>/knowledge before the fix
        ("../..", "exp"),
        ("../victim", "exp"),
        ("s1", "../../escape"),
    ],
)
def test_traversing_ids_are_refused_before_anything_is_deleted(
    tmp_path: Path, session_id: str, experiment_key: str
) -> None:
    """`research_branch_name` permits `.` and `/`, so `..` reaches the path.

    git rejects `..` in a refname — but only *after* `_force_unregister`'s
    rmtree has already run, so the containment check has to come first.
    """
    root = _repo(tmp_path)
    (root / WORKTREE_DIRNAME).mkdir()  # true after any prior experiment
    victim = root / "knowledge"
    victim.mkdir()
    (victim / "knowledge.db").write_text("hypotheses, beliefs, claims", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the experiment worktree root"):
        create_experiment_worktree(
            root, session_id=session_id, experiment_key=experiment_key
        )

    assert (victim / "knowledge.db").read_text() == "hypotheses, beliefs, claims"


@pytest.mark.parametrize(
    ("session_id", "experiment_key"),
    [
        (".", "."),  # research/./.  → resolves to .worktrees itself
        ("a", ".."),  # research/a/.. → same place by another route
    ],
)
def test_ids_resolving_to_the_worktree_root_itself_are_refused(
    tmp_path: Path, session_id: str, experiment_key: str
) -> None:
    """The guard's own boundary: *on* the root, not past it.

    An earlier version of the containment check tested `!= root and not
    is_relative_to(root)`. `is_relative_to` is already True for an equal path,
    so that first clause was dead and the root itself was permitted — and
    rmtree on the root destroys every concurrently running branch rather than
    one. Escaping outward was covered; landing exactly on the edge was not.
    """
    root = _repo(tmp_path)
    live_a = create_experiment_worktree(root, session_id="s1", experiment_key="branch-a")
    live_b = create_experiment_worktree(root, session_id="s1", experiment_key="branch-b")

    with pytest.raises(ValueError, match="worktree root"):
        create_experiment_worktree(
            root, session_id=session_id, experiment_key=experiment_key
        )

    # Every sibling branch survived.
    assert live_a.path.is_dir()
    assert live_b.path.is_dir()
    remove_experiment_worktree(live_a)
    remove_experiment_worktree(live_b)


def test_remove_refuses_a_handle_pointing_at_the_worktree_root(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    live = create_experiment_worktree(root, session_id="s1", experiment_key="branch-a")
    forged = ExperimentWorktree(
        path=experiment_worktree_root(root), branch="research/x/y", repo_root=root
    )

    with pytest.raises(ValueError, match="worktree root"):
        remove_experiment_worktree(forged)

    assert live.path.is_dir()
    remove_experiment_worktree(live)


def test_remove_refuses_a_handle_pointing_outside_the_worktree_root(
    tmp_path: Path,
) -> None:
    """`ExperimentWorktree` is a plain dataclass — the handle is not trusted."""
    root = _repo(tmp_path)
    victim = root / "knowledge"
    victim.mkdir()
    (victim / "knowledge.db").write_text("data", encoding="utf-8")
    forged = ExperimentWorktree(path=victim, branch="research/s1/x", repo_root=root)

    with pytest.raises(ValueError, match="outside the experiment worktree root"):
        remove_experiment_worktree(forged)

    assert (victim / "knowledge.db").exists()


def test_reconcile_does_not_report_a_removal_that_failed(tmp_path: Path) -> None:
    """A failed removal must not read as a reconciled one."""
    root = _repo(tmp_path)
    wt = create_experiment_worktree(root, session_id="s1", experiment_key="stuck")

    import labpilot.research_engine.agents.git_worktree as gw

    # Simulate a removal that cannot complete (read-only mount, busy file):
    # git's remove fails and the directory survives.
    original = gw._force_unregister
    gw._force_unregister = lambda tool, path: None
    try:
        result = reconcile_worktrees(root, live_branches=set())
    finally:
        gw._force_unregister = original

    assert wt.path.is_dir(), "precondition: the directory should still be there"
    assert wt.path not in result.removed, "reported a removal that did not happen"
    # The caller must be able to SEE the failure, not just find it in a log.
    assert wt.path in result.failed
    assert not result.ok
    remove_experiment_worktree(wt)


def test_create_is_idempotent_for_the_same_key(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    first = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")
    (first.path / "scratch.txt").write_text("x", encoding="utf-8")
    second = create_experiment_worktree(root, session_id="s1", experiment_key="exp-a")

    assert second.path == first.path
    assert second.branch == first.branch
    assert second.path.is_dir()
    remove_experiment_worktree(second)
