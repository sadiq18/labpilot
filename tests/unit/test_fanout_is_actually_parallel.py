"""K branches must overlap, and must be reachable from the CLI (M11 task 7).

The perf criterion in the task is "<1.5x baseline". Stated as wall-clock that
is a benchmark, and a flaky one on a shared CI box. Stated as overlap it is
the same claim and it is deterministic: K branches that each sleep for `d`
finish in well under `K*d` only if they actually ran at the same time. A
fan-out that silently serialised — a lock held across a branch, a barrier, an
event loop that never yields — is what this catches.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest
from typer.main import get_command
from helpers.cli import cli_runner

from labpilot.cli.conduct import conduct_app
from labpilot.research_engine.conductor.fanout import prepare_branches, run_branches
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace

_BRANCHES = 4
_HOLD_S = 0.20


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    client = scaffold_workspace(tmp_path / "titanic", "titanic")
    root = Path(client.root)
    for args in (
        ("init", "-q"),
        ("config", "user.email", "t@t"),
        ("config", "user.name", "t"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True)
    (root / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=root, check=True)
    return Workspace.from_client(client)


class _OverlapAgent:
    """Records how many branches were inside `execute` at the same time.

    Shaped like `ExperimentSpecialist`: the blocking work goes through
    `anyio.to_thread.run_sync`, because that offload is what makes branches
    overlap at all — see `test_an_agent_that_blocks_the_loop_gets_no_parallelism`.
    """

    def __init__(self, *, offload: bool = True) -> None:
        self.peak = 0
        self.offload = offload
        self._live = 0
        self._guard = threading.Lock()

    def _hold(self) -> None:
        with self._guard:
            self._live += 1
            self.peak = max(self.peak, self._live)
        try:
            # Blocking, not `anyio.sleep`: a training run occupies its thread,
            # and an `await` here would prove only that the loop interleaves.
            time.sleep(_HOLD_S)
        finally:
            with self._guard:
                self._live -= 1

    async def execute(self, task: Any, workspace: Workspace, context: Any) -> list[Any]:
        del task, workspace, context
        if self.offload:
            await anyio.to_thread.run_sync(self._hold)
        else:
            self._hold()
        return []


def test_branches_run_at_the_same_time_not_one_after_another(
    workspace: Workspace,
) -> None:
    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    ids = [
        store.create(
            observation=f"o{i}",
            reason=f"r{i}",
            prediction=f"p{i}",
            confidence=0.5,
            technique=f"t{i}",
        ).id
        for i in range(_BRANCHES)
    ]
    branches = prepare_branches(
        workspace,
        ids,
        session_id="S-1",
        repo_root=Path(workspace.root),
        make_plan=lambda _ws, h: f"P-{h}",
    )
    assert len(branches) == _BRANCHES
    agent = _OverlapAgent()

    started = time.perf_counter()
    outcomes = run_branches(
        branches,
        agent=agent,
        cohort_id="C-1",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {},
    )
    elapsed = time.perf_counter() - started

    assert all(o.ok for o in outcomes)
    assert agent.peak == _BRANCHES, (
        f"only {agent.peak} of {_BRANCHES} branches were ever running at once"
    )
    # Generous, because the assertion that matters is `peak` above — this is
    # here so a fan-out that overlaps but takes serial time still fails.
    assert elapsed < _HOLD_S * _BRANCHES * 0.75


def test_an_agent_that_blocks_the_loop_gets_no_parallelism(
    workspace: Workspace,
) -> None:
    """The constraint the fan-out rests on, stated out loud.

    `run_parallel_async` awaits `agent.execute` on one event loop, so an agent
    whose work blocks instead of offloading serialises every branch — K
    worktrees, K claims, K compute shares, and no concurrency, silently.
    `ExperimentSpecialist` is safe because task 9 moved its work to
    `anyio.to_thread.run_sync`; a future specialist that forgets to is not.
    """
    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    ids = [
        store.create(
            observation=f"o{i}",
            reason=f"r{i}",
            prediction=f"p{i}",
            confidence=0.5,
            technique=f"t{i}",
        ).id
        for i in range(2)
    ]
    branches = prepare_branches(
        workspace,
        ids,
        session_id="S-2",
        repo_root=Path(workspace.root),
        make_plan=lambda _ws, h: f"P-{h}",
    )
    agent = _OverlapAgent(offload=False)

    run_branches(
        branches,
        agent=agent,
        cohort_id="C-2",
        workspace=workspace,
        context=None,
        build_task=lambda branch, cohort: {},
    )

    assert agent.peak == 1


def _declared_options(command_name: str) -> set[str]:
    """Every option name a `conduct` subcommand declares.

    Read off the command's parameters rather than its rendered `--help`. Rich
    wraps the help panel to the terminal, and a narrow one truncates long option
    names — measured: `--branches` is present at COLUMNS=50 and gone at 40, so
    the rendered-text version of this passed on a laptop and failed on CI.
    """
    command = get_command(conduct_app).commands[command_name]
    return {opt for param in command.params for opt in param.opts}


def test_the_cli_exposes_the_fan_out_width() -> None:
    """Without a flag the whole feature is unreachable: `branches` defaults to
    1 everywhere, and nothing else sets it."""
    for command in ("run", "continue", "resume"):
        assert "--branches" in _declared_options(command), (
            f"{command} cannot ask for parallel branches"
        )


def test_the_cli_refuses_a_nonsense_width() -> None:
    result = cli_runner().invoke(conduct_app, ["run", "goal", "--branches", "0"])
    assert result.exit_code != 0
