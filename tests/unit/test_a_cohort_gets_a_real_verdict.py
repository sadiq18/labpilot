"""End to end: a real specialist's branches produce a cohort verdict (M11).

The gap this closes. Every other fan-out test uses a stub agent, and every
promotion test writes experiment records by hand, so nothing exercised the
chain the milestone exists to deliver:

    cohort_id in task metadata
      -> ExperimentSpecialist writes a record and emits ExperimentCompleted
      -> the promotion subscriber resolves the metric and direction
      -> a cohort file naming a winner

Which is how a wrong-direction verdict passed 2287 tests and four review
rounds: promotion asked the shared resolver without a `fallback_maximize` and
took its `True` default, so on `cv_rmse` it promoted 0.50 over 0.20 — the worse
branch. Nothing looked at a real verdict, so nothing noticed.

Runs the real `ExperimentSpecialist` through
`build_default_specialist_registry`, built once per test as
`_experiment_agent` builds it once per campaign, subscribers included — with
only the git snapshot and the training run stubbed. What ranks, and which way,
is entirely production code.

The branches run in sequence here; concurrent arrivals into one cohort file are
covered in `test_promotion_cohort.py` under the cohort lock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers.experiment_harness import (
    bundle,
    experiment_workspace,
    stub_experiment_io,
    training_task,
)

from labpilot.research_engine.agents.catalog import build_default_specialist_registry
from labpilot.research_engine.agents.facade import execute_agent_sync
from labpilot.research_engine.agents.promotion import cohort_path
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.workspace_facade import Workspace

_COHORT = "S-001-D-001"


def _profile(workspace: Workspace, direction: str) -> None:
    """A competition profile, the way a campaign that has run `analyze` has one.

    `paths.root` is the third of the three directories the direction lookup
    searches, and the only shared one — a branch's worktree is not guaranteed to
    carry a copy.
    """
    paths = ResearchPaths(workspace.knowledge_dir, workspace.competition).ensure()
    (paths.root / "competition.json").write_text(
        json.dumps({"metric": {"name": "cv_rmse", "direction": direction}}),
        encoding="utf-8",
    )


@pytest.fixture
def agent():
    """The one specialist every branch of a campaign runs.

    Built once, because `_experiment_agent` is called once per campaign — so all
    K branches publish to a single bus and a single subscriber set. A registry
    per branch would give each its own and quietly stop covering that.
    """
    registry = build_default_specialist_registry(dry_run_default=False)
    return registry.require("experiment").agent


def _run_branch(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    agent: Any,
    *,
    execution_id: str,
    metrics: dict[str, object],
) -> None:
    """One branch, through the real specialist and the real subscribers."""
    stub_experiment_io(monkeypatch, execution_id=execution_id, status="succeeded")
    # What a finished training run leaves behind for the specialist to read.
    (workspace.root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    execute_agent_sync(
        agent,
        training_task(plan_id=f"P-{execution_id}", cohort_id=_COHORT, dry_run=False),
        workspace,
        bundle(),
    )


def _verdict(workspace: Workspace) -> dict:
    path = cohort_path(workspace.effective_runs_dir, _COHORT)
    assert path.is_file(), "no cohort file — the branches were never compared"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = experiment_workspace(tmp_path)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    return ws


def test_the_lower_error_branch_wins_on_a_minimised_metric(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """The bug this file was written for. `cv_rmse` is an error metric, so 0.20
    beats 0.50 — and promotion has to learn that from the competition rather
    than from a default."""
    _profile(workspace, "minimize")

    _run_branch(workspace, monkeypatch, agent, execution_id="E-1", metrics={"cv_rmse": 0.50})
    _run_branch(workspace, monkeypatch, agent, execution_id="E-2", metrics={"cv_rmse": 0.20})

    state = _verdict(workspace)
    assert sorted(m["id"] for m in state["members"]) == ["E-1", "E-2"]
    assert state["metric_key"] == "cv_rmse"
    assert state["maximize"] is False
    assert state["promoted"] == "E-2"
    assert state["demoted"] == ["E-1"]


def test_the_higher_score_branch_wins_on_a_maximised_metric(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """The mirror, so a verdict that ignores the direction entirely cannot pass
    both halves."""
    _profile(workspace, "maximize")

    _run_branch(workspace, monkeypatch, agent, execution_id="E-1", metrics={"cv_rmse": 0.50})
    _run_branch(workspace, monkeypatch, agent, execution_id="E-2", metrics={"cv_rmse": 0.20})

    state = _verdict(workspace)
    assert state["maximize"] is True
    assert state["promoted"] == "E-1"
    assert state["demoted"] == ["E-2"]


def test_without_a_profile_the_cohort_records_members_but_no_winner(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """No profile means no direction, and promotion has none of its own to fall
    back on — the conductor's `BudgetConfig.maximize` is unreachable from
    `agents`. Declining is the fix for the wrong-winner bug: this is the exact
    case that used to promote 0.50 over 0.20.
    """
    _run_branch(workspace, monkeypatch, agent, execution_id="E-1", metrics={"cv_rmse": 0.50})
    _run_branch(workspace, monkeypatch, agent, execution_id="E-2", metrics={"cv_rmse": 0.20})

    state = _verdict(workspace)
    assert sorted(m["id"] for m in state["members"]) == ["E-1", "E-2"]
    assert state.get("promoted") is None
    assert state.get("metric_key") is None


def test_a_dry_run_cohort_records_members_but_promotes_nobody(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """`run_experiment` defaults to a dry run, and a dry run writes
    `status: dry_run_stub`. The placeholder guard refuses those, so the whole
    cohort is unrankable — correct on every individual rule, and worth pinning
    because the combination surprises: a dry-run fan-out spends K worktrees and
    can never produce a winner.
    """
    _profile(workspace, "minimize")

    for execution_id, score in (("E-1", 0.50), ("E-2", 0.20)):
        _run_branch(
            workspace,
            monkeypatch,
            agent,
            execution_id=execution_id,
            metrics={"status": "dry_run_stub", "cv_rmse": score},
        )

    state = _verdict(workspace)
    assert sorted(m["id"] for m in state["members"]) == ["E-1", "E-2"]
    assert state.get("promoted") is None


def test_the_cohort_survives_its_branch_workspaces_being_torn_down(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    """The verdict has to outlive the worktrees it compared. Written under
    `effective_runs_dir`, which `for_branch` pins to the shared workspace, so a
    cohort file is still readable once every branch's code root is gone."""
    _profile(workspace, "minimize")
    branch_root = workspace.root.parent / "worktrees" / "b1"
    branch_root.mkdir(parents=True)
    (branch_root / "pipeline").mkdir()
    branch = workspace.for_branch(branch_root)

    _run_branch(branch, monkeypatch, agent, execution_id="E-1", metrics={"cv_rmse": 0.50})
    _run_branch(branch, monkeypatch, agent, execution_id="E-2", metrics={"cv_rmse": 0.20})

    import shutil

    shutil.rmtree(branch_root)

    state = _verdict(workspace)
    assert state["promoted"] == "E-2"
    assert not cohort_path(workspace.effective_runs_dir, _COHORT).is_relative_to(
        branch_root
    ), "the verdict was written inside the branch it compared"
