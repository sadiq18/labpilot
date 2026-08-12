"""Setting up and tearing down K branches is a transaction (M11 task 7).

Every branch takes two things that outlive a failure — a hypothesis claim and
a checked-out worktree — so the interesting cases here are the ones where
something goes wrong part-way and the question is what got given back.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.fanout import (
    BranchOutcome,
    PlanRejected,
    prepare_branches,
    resolve_k,
    run_branches,
    select_hypotheses,
    teardown_branches,
)
from labpilot.research_engine.execution.training.compute_budget import (
    THREAD_LIMIT_VARS,
    cpu_share,
    thread_limit_env,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _repo(path: Path) -> Path:
    """A git repo with one commit — `worktree add` needs a HEAD to branch."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=path, check=True)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    _repo(Path(client.root))
    return Workspace.from_client(client)


def _propose(workspace: Workspace, *names: str) -> list[str]:
    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    return [
        store.create(
            observation=f"observed {name}",
            reason=f"because {name}",
            prediction=f"{name} improves the metric",
            confidence=0.5,
            technique=name,
        ).id
        for name in names
    ]


def _plans(counter: dict[str, int]):
    def make_plan(_ws: Workspace, hypothesis_id: str) -> str:
        counter[hypothesis_id] = counter.get(hypothesis_id, 0) + 1
        return f"P-{hypothesis_id}"

    return make_plan


class _RecordingAgent:
    """Stands in for `ExperimentSpecialist`: records what each branch was
    handed, and can fail for one of them."""

    def __init__(
        self,
        *,
        fail_for: set[Path] | None = None,
        execution_id: str = "E-001",
    ) -> None:
        self.fail_for = fail_for or set()
        self.execution_id = execution_id
        self.roots: list[Path] = []
        self.tasks: list[Any] = []
        self.envs: list[dict[str, str]] = []

    async def execute(self, task: Any, workspace: Workspace, context: Any) -> list[Any]:
        del context
        self.roots.append(workspace.root)
        self.tasks.append(task)
        # `thread_limit_env` rather than the contextvar: this is what a
        # training run is actually handed, so reading it here is the only
        # check that the cap survives all the way to the consumer.
        self.envs.append(thread_limit_env())
        if workspace.root in self.fail_for:
            raise RuntimeError("branch blew up")
        return [
            ArtifactRef(
                kind="experiment",
                id=f"experiment:{self.execution_id}",
                schema_id="experiment/v1",
            )
        ]


def _status(workspace: Workspace, hypothesis_id: str) -> HypothesisStatus:
    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    return store.get(hypothesis_id).status


# -- K resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        (1, 5, 1),  # not asked for
        (4, 1, 1),  # nothing to compare against
        (4, 0, 1),
        (0, 5, 1),
        (4, 2, 2),  # clamped by what exists
        (2, 9, 2),  # clamped by what was asked
        (3, 3, 3),
    ],
)
def test_k_is_clamped_by_both_the_request_and_what_exists(
    requested: int, available: int, expected: int
) -> None:
    assert resolve_k(requested, available=available) == expected


def test_one_hypothesis_is_never_a_fan_out() -> None:
    """K=1 must stay on the sequential path — a one-branch "fan-out" would
    take a worktree and a claim to do what the existing path already does."""
    assert resolve_k(8, available=1) == 1


# -- selection ------------------------------------------------------------


def test_selection_returns_the_top_hypotheses_and_no_more(workspace: Workspace) -> None:
    _propose(workspace, "a", "b", "c")
    assert len(select_hypotheses(workspace, 2)) == 2


def test_selection_is_empty_when_nothing_is_proposed(workspace: Workspace) -> None:
    assert select_hypotheses(workspace, 3) == []


def test_selection_ignores_hypotheses_already_being_tested(
    workspace: Workspace,
) -> None:
    ids = _propose(workspace, "a", "b")
    HypothesisStore(workspace.knowledge_dir, workspace.competition).claim_if_proposed(
        ids[0]
    )

    assert select_hypotheses(workspace, 5) == [ids[1]]


# -- preparation ----------------------------------------------------------


def test_each_prepared_branch_gets_its_own_worktree_and_workspace(
    workspace: Workspace, tmp_path: Path
) -> None:
    ids = _propose(workspace, "a", "b")

    branches = prepare_branches(
        workspace,
        ids,
        session_id="S-1",
        repo_root=Path(workspace.root),
        make_plan=_plans({}),
    )

    assert len(branches) == 2
    roots = {b.workspace.root for b in branches}
    assert len(roots) == 2, "branches must not share a code root"
    for branch in branches:
        assert branch.worktree.path.is_dir()
        assert branch.workspace.root == branch.worktree.path
        # Shared state stays shared — the whole point of `for_branch`.
        assert branch.workspace.knowledge_dir == workspace.knowledge_dir
        assert branch.workspace.data_dir == workspace.data_dir
        assert branch.workspace.effective_runs_dir == workspace.effective_runs_dir
        assert _status(workspace, branch.hypothesis_id) == HypothesisStatus.TESTING


def test_a_hypothesis_claimed_by_someone_else_is_skipped(
    workspace: Workspace,
) -> None:
    ids = _propose(workspace, "a", "b")
    HypothesisStore(workspace.knowledge_dir, workspace.competition).claim_if_proposed(
        ids[0]
    )
    planned: dict[str, int] = {}

    branches = prepare_branches(
        workspace,
        ids,
        session_id="S-1",
        repo_root=Path(workspace.root),
        make_plan=_plans(planned),
    )

    assert [b.hypothesis_id for b in branches] == [ids[1]]
    assert ids[0] not in planned, "a hypothesis we did not claim must not be planned"


def test_a_branch_whose_plan_fails_gives_its_claim_back(
    workspace: Workspace,
) -> None:
    """Otherwise the hypothesis sits in `testing` with nothing running it, and
    no later step will ever pick it up again."""
    ids = _propose(workspace, "a", "b")

    def make_plan(_ws: Workspace, hypothesis_id: str) -> str:
        if hypothesis_id == ids[0]:
            raise RuntimeError("planner is down")
        return f"P-{hypothesis_id}"

    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=make_plan
    )

    assert [b.hypothesis_id for b in branches] == [ids[1]]
    assert _status(workspace, ids[0]) == HypothesisStatus.PROPOSED
    assert _status(workspace, ids[1]) == HypothesisStatus.TESTING


def test_a_declined_plan_is_reported_as_a_decision_not_a_fault(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator using the approval gate is the gate working. Logged at ERROR
    with a traceback — which is what the broad `except Exception` did before
    `PlanRejected` existed — it trains operators to ignore the level that
    carries real faults, and buries a genuinely broken plan compiler among
    routine declines.
    """
    ids = _propose(workspace, "a")

    def decline(_ws: Workspace, hypothesis_id: str) -> str:
        raise PlanRejected(f"operator declined a plan for {hypothesis_id}")

    with caplog.at_level(logging.INFO, logger="labpilot.research_engine.conductor.fanout"):
        branches = prepare_branches(
            workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=decline
        )

    assert branches == []
    assert _status(workspace, ids[0]) == HypothesisStatus.PROPOSED
    declines = [r for r in caplog.records if "declined" in r.getMessage()]
    assert declines, "the decline was not reported at all"
    assert all(r.levelno == logging.INFO for r in declines), (
        f"a decline was logged at {[r.levelname for r in declines]}"
    )
    assert all(r.exc_info is None for r in declines), "a decline carried a traceback"


def test_a_genuine_planning_failure_is_still_an_error(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """The other half — quietening declines must not quieten faults. A broken
    plan compiler is exactly what the ERROR level is for."""
    ids = _propose(workspace, "a")

    def explode(_ws: Workspace, _hypothesis_id: str) -> str:
        raise RuntimeError("planner is down")

    with caplog.at_level(logging.INFO, logger="labpilot.research_engine.conductor.fanout"):
        prepare_branches(
            workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=explode
        )

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a real planning failure was swallowed"
    assert any(r.exc_info is not None for r in errors), "the fault lost its traceback"


def test_a_branch_whose_worktree_fails_gives_its_claim_back(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _propose(workspace, "a")
    monkeypatch.setattr(
        "labpilot.research_engine.conductor.fanout.create_experiment_worktree",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("already checked out")),
    )

    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )

    assert branches == []
    assert _status(workspace, ids[0]) == HypothesisStatus.PROPOSED


# -- teardown -------------------------------------------------------------


def test_teardown_removes_every_worktree(workspace: Workspace) -> None:
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    outcomes = [
        BranchOutcome(hypothesis_id=b.hypothesis_id, plan_id=b.plan_id, ok=True)
        for b in branches
    ]

    teardown_branches(workspace, branches, outcomes)

    assert all(not b.worktree.path.exists() for b in branches)


def test_a_failed_branch_returns_its_hypothesis_to_the_pool(
    workspace: Workspace,
) -> None:
    """A failure produced no answer, so the hypothesis is still untested. A
    success keeps its claim — the reflection its own run files is what
    resolves it, and handing it back would let the next step re-test a
    question that already has an answer."""
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    outcomes = [
        BranchOutcome(hypothesis_id=branches[0].hypothesis_id, plan_id="P-1", ok=False),
        BranchOutcome(hypothesis_id=branches[1].hypothesis_id, plan_id="P-2", ok=True),
    ]

    teardown_branches(workspace, branches, outcomes)

    assert _status(workspace, branches[0].hypothesis_id) == HypothesisStatus.PROPOSED
    assert _status(workspace, branches[1].hypothesis_id) == HypothesisStatus.TESTING


def test_running_branches_gives_each_agent_its_own_workspace(
    workspace: Workspace,
) -> None:
    """The per-item workspace is the whole isolation mechanism: without it
    `run_parallel_async` hands every branch the caller's one, and K branches
    write their code into the shared workspace the worktrees exist to protect.
    """
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    agent = _RecordingAgent()

    outcomes = run_branches(
        branches,
        agent=agent,
        cohort_id="C-1",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {"plan_id": branch.plan_id, "cohort_id": cohort},
    )

    assert all(o.ok for o in outcomes)
    assert sorted(agent.roots) == sorted(b.worktree.path for b in branches)
    assert workspace.root not in agent.roots


def test_running_branches_divides_the_machine_and_restores_it(
    workspace: Workspace,
) -> None:
    """`cpu_share(K)` must be live while the branches run and gone after, or
    the next sequential step trains on a fraction of the machine."""
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    agent = _RecordingAgent()

    assert thread_limit_env() == {}, "the cap must not already be installed"

    run_branches(
        branches,
        agent=agent,
        cohort_id="C-1",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {},
    )

    assert thread_limit_env() == {}, "the cap outlived the fan-out"
    share = cpu_share(len(branches))
    # `cpu_share` returns None where the machine's size is undiscoverable, and
    # then there is no cap to observe — assert the share that was computed,
    # whichever it is, rather than assuming this machine reports its CPUs.
    expected = {name: str(share) for name in THREAD_LIMIT_VARS} if share else {}
    assert agent.envs == [expected, expected]


def test_one_failing_branch_does_not_take_its_siblings_down(
    workspace: Workspace,
) -> None:
    ids = _propose(workspace, "a", "b", "c")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    agent = _RecordingAgent(fail_for={branches[1].worktree.path})

    outcomes = run_branches(
        branches,
        agent=agent,
        cohort_id="C-1",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {},
    )

    by_id = {o.hypothesis_id: o for o in outcomes}
    assert by_id[branches[0].hypothesis_id].ok
    assert not by_id[branches[1].hypothesis_id].ok
    assert "branch blew up" in (by_id[branches[1].hypothesis_id].error or "")
    assert by_id[branches[2].hypothesis_id].ok


def test_a_branchs_execution_id_is_read_off_its_refs(workspace: Workspace) -> None:
    """A `ParallelResult` carries no `ToolResult.data`, so the id has to come
    from the ref. Nothing else can record the branch's score against it, and
    the loss would be silent."""
    ids = _propose(workspace, "a")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    agent = _RecordingAgent(execution_id="E-042")

    outcomes = run_branches(
        branches,
        agent=agent,
        cohort_id="C-1",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {},
    )

    assert outcomes[0].execution_id == "E-042"


def test_the_cohort_id_reaches_every_branchs_task(workspace: Workspace) -> None:
    """Without it `ExperimentSpecialist` emits no cohort_id, the promotion
    subscriber returns early, and nothing compares the branches at all."""
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    agent = _RecordingAgent()

    run_branches(
        branches,
        agent=agent,
        cohort_id="C-7",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {"cohort_id": cohort, "plan_id": branch.plan_id},
    )

    assert [t["cohort_id"] for t in agent.tasks] == ["C-7", "C-7"]
    assert sorted(t["plan_id"] for t in agent.tasks) == sorted(b.plan_id for b in branches)


def test_teardown_with_no_outcomes_releases_every_claim(
    workspace: Workspace,
) -> None:
    """The shape the production fallback actually passes: branches prepared,
    outcomes empty — when too few branches could be set up, or when the
    fan-out raised before producing any.

    Deriving the release set from the *failures* released nobody here, so every
    worktree was deleted while every hypothesis stayed `testing`. Selection
    only ever lists `proposed`, so those hypotheses became permanently
    unreachable by any later step.
    """
    ids = _propose(workspace, "a", "b")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    assert all(_status(workspace, i) == HypothesisStatus.TESTING for i in ids)

    teardown_branches(workspace, branches, [])

    assert [_status(workspace, i) for i in ids] == [
        HypothesisStatus.PROPOSED,
        HypothesisStatus.PROPOSED,
    ]
    assert all(not b.worktree.path.exists() for b in branches)


def test_a_single_prepared_branch_is_released_when_the_fan_out_backs_out(
    workspace: Workspace,
) -> None:
    """The likeliest route into the bug above: one claimable hypothesis, so
    the caller tears down and falls back to the sequential path. If the claim
    were not returned, the sequential path could not test it either."""
    ids = _propose(workspace, "only-one")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )

    teardown_branches(workspace, branches, [])

    assert _status(workspace, ids[0]) == HypothesisStatus.PROPOSED


def test_a_branch_whose_worktree_will_not_go_keeps_its_claim(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Releasing a hypothesis whose branch is still checked out makes it
    selectable again, and every later step then claims it, fails
    `worktree add` with "already checked out", and releases it — burning a
    fan-out slot each time. Left claimed, it is simply skipped."""
    ids = _propose(workspace, "stuck", "fine")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    real_remove = teardown_branches.__globals__["remove_experiment_worktree"]

    def flaky(worktree: Any, **kw: Any) -> None:
        if worktree.path == branches[0].worktree.path:
            raise RuntimeError("directory busy")
        real_remove(worktree, **kw)

    monkeypatch.setattr(
        "labpilot.research_engine.conductor.fanout.remove_experiment_worktree", flaky
    )

    teardown_branches(workspace, branches, [])

    assert _status(workspace, branches[0].hypothesis_id) == HypothesisStatus.TESTING
    assert _status(workspace, branches[1].hypothesis_id) == HypothesisStatus.PROPOSED


def test_one_unremovable_worktree_does_not_strand_the_others(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worktree left registered keeps its branch checked out, so the next
    run of that experiment key cannot claim it — stranding the rest turns one
    stuck branch into K stuck branches."""
    ids = _propose(workspace, "a", "b", "c")
    branches = prepare_branches(
        workspace, ids, session_id="S-1", repo_root=Path(workspace.root), make_plan=_plans({})
    )
    real_remove = teardown_branches.__globals__["remove_experiment_worktree"]

    def flaky(worktree: Any, **kw: Any) -> None:
        if worktree.path == branches[0].worktree.path:
            raise RuntimeError("directory busy")
        real_remove(worktree, **kw)

    monkeypatch.setattr(
        "labpilot.research_engine.conductor.fanout.remove_experiment_worktree", flaky
    )

    teardown_branches(workspace, branches, [])

    assert branches[0].worktree.path.exists()
    assert not branches[1].worktree.path.exists()
    assert not branches[2].worktree.path.exists()
