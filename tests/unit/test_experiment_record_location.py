"""M11: the experiment record survives its branch's root being torn down.

Once K-way fan-out (task 7) roots each branch's `workspace.root` at its own
worktree, anything written there is private to that worktree and gone at
teardown. The record has to live somewhere every branch of one campaign
agrees on regardless of `root` — `workspace.effective_runs_dir` — or the
promotion subscriber (task 6) has nothing to compare siblings against once
any one of them tears its worktree down.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import anyio
import pytest
from helpers.experiment_harness import (
    bundle,
    experiment_workspace,
    stub_experiment_io,
    training_task,
)

from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.git_evolution import find_experiment_record
from labpilot.research_engine.workspace_facade import Workspace


def _worktree_shaped(tmp_path: Path, name: str) -> Workspace:
    """A workspace whose `root` is private but whose `runs_dir` is shared.

    Mirrors what task 7 builds per branch: `code_root` (here, `root`) unique
    per call, `runs_dir` pinned to the same shared directory across calls —
    the split PR #136 and this task both depend on.
    """
    shared = experiment_workspace(tmp_path)
    branch_root = tmp_path / "branches" / name
    branch_root.mkdir(parents=True)
    return shared.model_copy(update={"root": branch_root, "runs_dir": shared.effective_runs_dir})


def test_the_record_is_findable_after_its_branch_root_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = _worktree_shaped(tmp_path, "E-1")
    branch_root = ws.root

    anyio.run(lambda: ExperimentSpecialist().execute(training_task(), ws, bundle()))

    # The branch's own root — the worktree — is gone, the way task 3's
    # teardown removes it once the branch finishes.
    shutil.rmtree(branch_root)

    record = find_experiment_record(ws.effective_runs_dir, "E-1")
    assert record is not None
    assert record["execution_id"] == "E-1"


def test_two_branches_with_different_roots_share_one_record_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the promotion subscriber (task 6) depends on directly:
    siblings with different `root`s must still land in the one place it
    looks for cohort members.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    branch_a = _worktree_shaped(tmp_path, "E-1")
    anyio.run(lambda: ExperimentSpecialist().execute(training_task(), branch_a, bundle()))

    stub_experiment_io(monkeypatch, execution_id="E-2", status="succeeded")
    branch_b = _worktree_shaped(tmp_path, "E-2")
    anyio.run(lambda: ExperimentSpecialist().execute(training_task(), branch_b, bundle()))

    assert branch_a.root != branch_b.root
    assert branch_a.effective_runs_dir == branch_b.effective_runs_dir
    for execution_id in ("E-1", "E-2"):
        record = find_experiment_record(branch_a.effective_runs_dir, execution_id)
        assert record is not None
        assert record["execution_id"] == execution_id


def test_event_payload_carries_the_shared_runs_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer reconstructing paths from the event must have an explicit
    pointer to the shared location — deriving it from `workspace_root` alone
    gives the wrong answer once that field is a per-branch worktree path.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = _worktree_shaped(tmp_path, "E-1")
    seen: list[tuple[str, dict[str, Any]]] = []

    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    anyio.run(lambda: agent.execute(training_task(), ws, bundle()))

    ((_, payload),) = seen
    assert payload["runs_dir"] == str(ws.effective_runs_dir)
    assert payload["runs_dir"] != payload["workspace_root"]


def test_cohort_id_is_propagated_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = experiment_workspace(tmp_path)
    seen: list[tuple[str, dict[str, Any]]] = []

    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    anyio.run(
        lambda: agent.execute(training_task(cohort_id="step-42"), ws, bundle())
    )

    ((_, payload),) = seen
    assert payload["cohort_id"] == "step-42"
    record = find_experiment_record(ws.effective_runs_dir, "E-1")
    assert record is not None
    assert record["cohort_id"] == "step-42"


def test_cohort_id_is_absent_without_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The common, non-fan-out case must not carry a stray `cohort_id: None`
    — a promotion subscriber checking `payload.get("cohort_id")` truthiness
    needs the key genuinely missing, not present-and-falsy, to read cleanly
    either way, but a present key invites a reader to assume it's meaningful.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = experiment_workspace(tmp_path)
    seen: list[tuple[str, dict[str, Any]]] = []

    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    anyio.run(lambda: agent.execute(training_task(), ws, bundle()))

    ((_, payload),) = seen
    assert "cohort_id" not in payload
    record = find_experiment_record(ws.effective_runs_dir, "E-1")
    assert record is not None
    assert "cohort_id" not in record


def test_cohort_id_survives_a_failed_run_unlike_completed_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`event_payload` is shared with `ModelFailed`. `completed_at` is
    withheld on that path deliberately — a finish time on a run that died
    would assert a completion that never happened. `cohort_id` is not
    withheld: it identifies which step's fan-out produced this branch, which
    is true and useful whether the branch succeeded or not, so it's added
    to the dict before the success/failure branch splits rather than after.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="failed", error="boom")
    ws = experiment_workspace(tmp_path)
    seen: list[tuple[str, dict[str, Any]]] = []

    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    anyio.run(
        lambda: agent.execute(training_task(cohort_id="step-42"), ws, bundle())
    )

    ((event, payload),) = seen
    assert event == "ModelFailed"
    assert payload["cohort_id"] == "step-42"
    assert "completed_at" not in payload
