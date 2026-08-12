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
    monkeypatch.setattr(
        "labpilot.research_engine.conductor.loop._experiment_agent",
        lambda **kw: agent,
    )
    return _fan_out_experiment(
        store,
        workspace,
        session_id,
        step=3,
        branches=branches,
        rationale="test the top hypotheses",
        llm_client=None,
        dry_run=True,
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
        assert len(decisions) == 2
        assert len({d.id for d in decisions}) == 2, "branch decisions must be distinct"
        cohorts = {d.observe["cohort_id"] for d in decisions}
        assert cohorts == {f"{session.id}-step3"}
        assert {d.observe["branch"] for d in decisions} == {
            d.args["hypothesis_id"] for d in decisions
        }
        assert len(store.list_decisions(session.id)) == 2


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


def test_untracking_survives_a_workspace_that_is_not_a_repo(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "plain", "plain")
    _untrack_shared_state(Workspace.from_client(client))  # must not raise
