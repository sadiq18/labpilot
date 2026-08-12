"""The conductor's side of a fan-out (M11 task 7).

Covers what `test_fanout_branches.py` cannot: that each branch gets its own
audit record and its own entry in the circuit breaker's counters, and that
asking for one branch leaves the sequential path exactly as it was.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from labpilot.research_engine.conductor.checkpoint import load_budget_pair
from labpilot.research_engine.conductor.loop import (
    _DRY_RUN_DEFAULTS,
    _as_pathspecs,
    _fan_out_experiment,
    _untrack_shared_state,
)
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import SHARED_STATE_IGNORES, scaffold_workspace


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    root = Path(client.root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "seed")
    return Workspace.from_client(client)


def _propose(workspace: Workspace, *names: str) -> list[str]:
    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    return [
        store.create(
            observation=f"observed {n}",
            reason=f"because {n}",
            prediction=f"{n} helps",
            confidence=0.5,
            technique=n,
        ).id
        for n in names
    ]


class _Agent:
    """An experiment specialist that succeeds or fails per branch."""

    def __init__(self, fail_plans: set[str] | None = None) -> None:
        self.fail_plans = fail_plans or set()

    async def execute(self, task: Any, workspace: Workspace, context: Any) -> list[Any]:
        del workspace, context
        if task.metadata["plan_id"] in self.fail_plans:
            raise RuntimeError("training diverged")
        return []


def _fan_out(
    store: ConductorStore,
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: str,
    agent: _Agent,
    branches: int = 2,
) -> Any:
    """Run a fan-out with plan compilation and the specialist stubbed out."""
    del monkeypatch  # the agent is injected now, not patched in
    return _fan_out_experiment(
        store,
        workspace,
        session_id,
        step=3,
        branches=branches,
        rationale="test the top hypotheses",
        llm_client=None,
        dry_run=True,
        submit=False,
        agent=agent,
        progress=lambda _m: None,
    )


def test_every_branch_gets_its_own_decision_record(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One decision for K experiments would make the audit log unable to say
    which branch did what."""
    _propose(workspace, "a", "b")
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan",
        lambda ws, **kw: type("R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}})(),
    )
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("beat baseline")

        decisions = _fan_out(
            store, workspace, monkeypatch, session_id=session.id, agent=_Agent()
        )

        assert decisions is not None
        # One record naming the fan-out, then one per branch.
        assert [d.tool_name for d in decisions] == [
            "fan_out",
            "run_experiment",
            "run_experiment",
        ]
        assert len({d.id for d in decisions}) == 3, "decisions must be distinct"
        branch_records = decisions[1:]
        cohorts = {d.observe["cohort_id"] for d in branch_records}
        assert len(cohorts) == 1, "branches of one step must share one cohort"
        assert cohorts.pop() == f"{session.id}-{decisions[0].id}"
        assert {d.observe["branch"] for d in branch_records} == {
            d.args["hypothesis_id"] for d in branch_records
        }
        assert len(store.list_decisions(session.id)) == 3


def test_two_fan_outs_in_one_session_never_share_a_cohort(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`new_decision_id()` computes MAX+1 and reserves nothing, so naming the
    cohort with a peeked id handed the next fan-out the same one whenever the
    previous had recorded no decisions — merging two cohorts' members into one
    verdict, the same bug the step-number key had.

    The discriminating case is a fan-out that records nothing because it
    raised. One that completes advances MAX(id) either way and cannot tell the
    two implementations apart.
    """
    _plan_ok = lambda ws, **kw: type(  # noqa: E731
        "R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}}
    )()
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan", _plan_ok
    )

    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("beat baseline")

        _propose(workspace, "a0", "b0")
        monkeypatch.setattr(
            "labpilot.research_engine.conductor.fanout.run_parallel_sync",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("worker pool died")),
        )
        with pytest.raises(RuntimeError):
            _fan_out(store, workspace, monkeypatch, session_id=session.id, agent=_Agent())

        after_failure = store.list_decisions(session.id)
        assert [d.tool_name for d in after_failure] == ["fan_out"], (
            "the fan-out must name itself before running, or the id it claimed "
            "is not reserved against the next one"
        )
        first_cohort = f"{session.id}-{after_failure[-1].id}"

        monkeypatch.setattr(
            "labpilot.research_engine.conductor.fanout.run_parallel_sync",
            _real_run_parallel_sync(),
        )
        _propose(workspace, "a1", "b1")
        decisions = _fan_out(
            store, workspace, monkeypatch, session_id=session.id, agent=_Agent()
        )
        assert decisions is not None
        second_cohort = decisions[1].observe["cohort_id"]

    assert second_cohort != first_cohort, (
        f"a fan-out that recorded nothing handed its cohort id on: {first_cohort}"
    )


def test_a_store_failure_while_naming_the_cohort_still_tears_down(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branches exist by the time the cohort is named, so everything from
    that point on has to be inside the teardown's `try`. Naming the cohort
    writes to the store and can fail; above the `try` it would leave K claims
    held and K worktrees checked out with nothing to release them.
    """
    ids = _propose(workspace, "a", "b")
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan",
        lambda ws, **kw: type("R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}})(),
    )
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("beat baseline")
        monkeypatch.setattr(
            type(store),
            "append_new_decision",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db is gone")),
        )

        with pytest.raises(RuntimeError):
            _fan_out(store, workspace, monkeypatch, session_id=session.id, agent=_Agent())

    hypotheses = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    assert [hypotheses.get(i).status for i in ids] == [
        HypothesisStatus.PROPOSED,
        HypothesisStatus.PROPOSED,
    ]
    assert ".worktrees" not in _git(Path(workspace.root), "worktree", "list")


def _real_run_parallel_sync() -> Any:
    from labpilot.research_engine.agents.parallel import run_parallel_sync

    return run_parallel_sync


def test_each_branch_feeds_the_circuit_breaker_separately(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counting K failures as one lets a campaign run past the point the
    breaker exists to stop it at — the exact defect measured on 2026-08-09,
    one layer up."""
    ids = _propose(workspace, "a", "b", "c")
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan",
        lambda ws, **kw: type("R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}})(),
    )
    agent = _Agent(fail_plans={f"P-{i}" for i in ids})

    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("beat baseline")

        _fan_out(
            store, workspace, monkeypatch, session_id=session.id, agent=agent, branches=3
        )

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.consecutive_failures == 3


def test_a_successful_fan_out_resets_the_breaker(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _propose(workspace, "a", "b")
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan",
        lambda ws, **kw: type("R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}})(),
    )
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("beat baseline")

        _fan_out(store, workspace, monkeypatch, session_id=session.id, agent=_Agent())

        _, state = load_budget_pair(store.get_session(session.id))
        assert state.consecutive_failures == 0


def test_asking_for_one_branch_is_not_a_fan_out(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`None` means "I did nothing, dispatch normally" — the guarantee that
    the K=1 path is untouched."""
    _propose(workspace, "a", "b")
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("g")
        assert (
            _fan_out(
                store,
                workspace,
                monkeypatch,
                session_id=session.id,
                agent=_Agent(),
                branches=1,
            )
            is None
        )


def test_a_single_untested_hypothesis_is_not_a_fan_out(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to compare against, so the branch machinery buys nothing and
    the hypothesis must be left unclaimed for the sequential path."""
    ids = _propose(workspace, "only-one")
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("g")

        assert (
            _fan_out(store, workspace, monkeypatch, session_id=session.id, agent=_Agent())
            is None
        )

    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    assert store.get(ids[0]).status == HypothesisStatus.PROPOSED


def test_a_fan_out_leaves_no_worktrees_behind(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _propose(workspace, "a", "b")
    monkeypatch.setattr(
        "labpilot.research_engine.tools.handlers.plan.generate_plan",
        lambda ws, **kw: type("R", (), {"data": {"plan_id": f"P-{kw['hypothesis_id']}"}})(),
    )
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("g")
        _fan_out(store, workspace, monkeypatch, session_id=session.id, agent=_Agent())

    registered = _git(Path(workspace.root), "worktree", "list")
    assert ".worktrees" not in registered


# -- startup reconciliation -----------------------------------------------


def test_a_running_campaigns_worktrees_are_not_swept(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two campaigns on one workspace is ordinary. A sweep that assumed
    nothing else was live would delete a running campaign's checkouts out
    from under its branches, mid-experiment."""
    from labpilot.research_engine.agents.git_worktree import create_experiment_worktree
    from labpilot.research_engine.conductor.loop import _reconcile_stale_worktrees

    root = Path(workspace.root)
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        live = store.create_session("still going")
        finished = store.create_session("all done")
        store.update_session_status(finished.id, "completed")

        live_tree = create_experiment_worktree(
            root, session_id=live.id, experiment_key="H-live"
        )
        dead_tree = create_experiment_worktree(
            root, session_id=finished.id, experiment_key="H-dead"
        )
        assert live_tree.path.is_dir() and dead_tree.path.is_dir()

        _reconcile_stale_worktrees(store, workspace)

    assert live_tree.path.is_dir(), "a running campaign lost its worktree"
    assert not dead_tree.path.exists(), "a finished campaign's worktree was kept"


def test_another_competitions_worktrees_are_left_alone(
    workspace: Workspace, tmp_path: Path
) -> None:
    """`list_sessions()` filters by competition and every competition keeps its
    own knowledge.db, so a sibling campaign's session is not merely absent from
    the live set — it is unreadable from here. Sweeping on "not known means
    dead" would delete its worktrees mid-experiment."""
    from labpilot.research_engine.agents.git_worktree import create_experiment_worktree
    from labpilot.research_engine.conductor.loop import _reconcile_stale_worktrees

    root = Path(workspace.root)
    # A worktree whose session id this store has never heard of.
    foreign = create_experiment_worktree(
        root, session_id="S-999", experiment_key="H-foreign"
    )
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        mine = store.create_session("mine")
        store.update_session_status(mine.id, "completed")
        stale = create_experiment_worktree(
            root, session_id=mine.id, experiment_key="H-mine"
        )

        _reconcile_stale_worktrees(store, workspace)

    assert foreign.path.is_dir(), "another competition's worktree was swept"
    assert not stale.path.exists(), "our own finished worktree was kept"


def test_a_paused_campaign_keeps_its_worktrees(
    workspace: Workspace,
) -> None:
    """Resuming a paused campaign needs its branches still checked out —
    sweeping them turns a pause into a silent loss of work."""
    from labpilot.research_engine.agents.git_worktree import create_experiment_worktree
    from labpilot.research_engine.conductor.loop import _reconcile_stale_worktrees

    root = Path(workspace.root)
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        paused = store.create_session("held")
        store.update_session_status(paused.id, "paused")
        tree = create_experiment_worktree(
            root, session_id=paused.id, experiment_key="H-1"
        )

        _reconcile_stale_worktrees(store, workspace)

    assert tree.path.is_dir()


def test_an_unreadable_session_table_sweeps_nothing(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not knowing which campaigns are live is a reason to delete nothing: a
    stale worktree costs the next sweep, a deleted one costs a running run."""
    from labpilot.research_engine.agents.git_worktree import create_experiment_worktree
    from labpilot.research_engine.conductor.loop import _reconcile_stale_worktrees

    root = Path(workspace.root)
    with ConductorStore(workspace.knowledge_dir, workspace.competition) as store:
        session = store.create_session("g")
        store.update_session_status(session.id, "completed")
        tree = create_experiment_worktree(root, session_id=session.id, experiment_key="H-1")
        monkeypatch.setattr(
            type(store),
            "list_sessions",
            lambda self: (_ for _ in ()).throw(RuntimeError("db gone")),
        )

        _reconcile_stale_worktrees(store, workspace)

    assert tree.path.is_dir()


# -- the index migration --------------------------------------------------


def test_pathspecs_drop_the_gitignore_anchor() -> None:
    """A leading `/` anchors a gitignore pattern but means an absolute path to
    a pathspec — git rejects `/runs/` with `fatal: Invalid path '/runs'`."""
    assert _as_pathspecs(("/runs/", "/knowledge/**/knowledge.db")) == [
        "runs/",
        "knowledge/**/knowledge.db",
    ]
    assert all(not p.startswith("/") for p in _as_pathspecs(SHARED_STATE_IGNORES))


def test_already_tracked_shared_state_is_untracked(workspace: Workspace) -> None:
    """A `.gitignore` pattern does not untrack an existing file, so a
    workspace scaffolded before the pattern keeps copying its database into
    every worktree — measured at 105 MB per branch."""
    root = Path(workspace.root)
    db = root / "knowledge" / "research" / "knowledge.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text("db", encoding="utf-8")
    runs = root / "runs" / "E-1"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "out.json").write_text("{}", encoding="utf-8")
    _git(root, "add", "-A", "-f")
    _git(root, "commit", "-qm", "pre-migration workspace")
    assert "knowledge.db" in _git(root, "ls-files")

    _untrack_shared_state(workspace)

    tracked = _git(root, "ls-files")
    assert "knowledge.db" not in tracked
    assert "runs/" not in tracked
    assert "seed.txt" in tracked, "unrelated files must stay tracked"
    assert db.is_file(), "--cached must not delete the database from disk"


def test_untracking_is_a_no_op_when_nothing_is_tracked(workspace: Workspace) -> None:
    before = _git(Path(workspace.root), "ls-files")

    _untrack_shared_state(workspace)

    assert _git(Path(workspace.root), "ls-files") == before


def test_untracking_does_not_create_a_repo_where_there_is_none(
    tmp_path: Path,
) -> None:
    """`open_git_tool` runs `Repo.init()` plus a bootstrap commit on a
    directory that is not a repository, so reaching for it before checking
    created one in a workspace the user had deliberately kept out of version
    control — the same side effect `reconcile_worktrees` refuses, from the same
    unattended startup path.

    Asserting the absence, not just that nothing raised: creating a repo does
    not raise, so the earlier "must not raise" test was green while the bug
    happened.
    """
    client = scaffold_workspace(tmp_path / "plain", "plain")
    root = Path(client.root)
    assert not (root / ".git").exists()

    _untrack_shared_state(Workspace.from_client(client))

    assert not (root / ".git").exists(), "a git repository was created"


def test_the_fan_out_dry_run_default_matches_the_tool_it_replaces() -> None:
    """`run_plan` trains for real unless told otherwise; `run_experiment` does
    not. Hardcoding either default made the other wrong — defaulting to a dry
    run turned an unset `run_plan` fan-out into K branches that trained
    nothing and, once the placeholder guard refused their stub metrics,
    promoted nobody."""
    from labpilot.research_engine.tools.handlers import run as run_mod

    assert _DRY_RUN_DEFAULTS["run_plan"] == _signature_default(run_mod.run_plan, "dry_run")
    assert _DRY_RUN_DEFAULTS["run_experiment"] == _signature_default(
        _run_experiment_handler(), "dry_run"
    )
    assert set(_DRY_RUN_DEFAULTS) == {"run_plan", "run_experiment"}, (
        "a new experiment tool needs its own default here, or it silently "
        "inherits the wrong one"
    )


def _signature_default(fn: Any, name: str) -> Any:
    import inspect

    return inspect.signature(fn).parameters[name].default


def _run_experiment_handler() -> Any:
    from labpilot.research_engine.tools.handlers.specialists import run_experiment

    return run_experiment
