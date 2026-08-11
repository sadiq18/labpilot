"""M11 task 5: `ExperimentCompleted` carries when the run finished.

Promotion (task 6) ranks branches on the metric and breaks a tie by earliest
finisher, so the payload needs a finish time it does not have today. The
stamp is only meaningful on a run that actually finished: the failure path
shares this dict, and a `completed_at` on a run that died would assert the
completion `test_failed_run_is_not_completed.py` exists to deny.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest

from labpilot.research_engine.agents import experiment as experiment_mod
from labpilot.research_engine.agents.events import EXPERIMENT_COMPLETED, MODEL_FAILED
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.models import AgentTask
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def _ws(tmp_path: Path) -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", "demo", code_root=tmp_path / "ws"
    ).ensure_roots()


def _bundle() -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition="demo", goal="test"))


@pytest.fixture
def stub_run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Run `execute` without git branching or a real training run.

    Returns a setter for the outcome `run_plan` reports. Both patches are
    installed now and the setter only replaces one of them, so a test that
    takes this fixture and forgets to call the setter still cannot reach the
    real plan runner — it gets an empty result instead of doing real work.
    """
    monkeypatch.setattr(experiment_mod, "snapshot_before_experiment", lambda *a, **k: None)

    def _set(**data: Any) -> None:
        def _fake_run_plan(*_a: object, **_k: object) -> ToolResult:
            # The real handler is annotated `-> ToolResult`; returning one
            # keeps the stub honest about the contract `execute` consumes.
            return ToolResult(data=dict(data), refs=[])

        monkeypatch.setattr("labpilot.research_engine.tools.handlers.run.run_plan", _fake_run_plan)

    _set()
    return _set


def _run(tmp_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Execute the specialist, returning every event it emitted."""
    seen: list[tuple[str, dict[str, Any]]] = []
    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    task = AgentTask(id="T-1", capability="run_training", description="train")

    anyio.run(lambda: agent.execute(task, _ws(tmp_path), _bundle()))
    return seen


def test_a_completed_run_is_stamped_with_when_it_finished(
    tmp_path: Path, stub_run: Callable[..., None]
) -> None:
    """The stamp must be this run's finish time, not any constant.

    Bounding it by clock reads taken either side of the call is what makes
    that assertion — a hardcoded or stale value passes a mere "is a
    timestamp" check and fails this one.
    """
    stub_run(execution_id="E-1", status="succeeded")

    before = datetime.now(UTC)
    events = _run(tmp_path)
    after = datetime.now(UTC)

    # Asserting the whole event list rather than filtering for the one wanted:
    # emitting ModelFailed *as well* would otherwise pass unnoticed.
    assert [e for e, _ in events] == [EXPERIMENT_COMPLETED]
    ((_, payload),) = events
    stamped = datetime.fromisoformat(payload["completed_at"])
    assert stamped.tzinfo is not None, "a naive stamp cannot be compared across hosts"
    assert before <= stamped <= after


def test_a_failed_run_carries_no_completion_time(
    tmp_path: Path, stub_run: Callable[..., None]
) -> None:
    """The load-bearing half: `event_payload` is shared with `ModelFailed`.

    Stamping the dict literal instead of the success path would put a finish
    time on a run that crashed — the same false completion that published a
    six-day-old rmse and cost sixteen dispatches.
    """
    stub_run(execution_id="E-147", status="failed", error="No module named 'catboost'")

    events = _run(tmp_path)

    assert [e for e, _ in events] == [MODEL_FAILED]
    ((_, payload),) = events
    assert "completed_at" not in payload


def test_the_stamp_excludes_the_record_write(
    tmp_path: Path, stub_run: Callable[..., None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp times the run, not the bookkeeping that follows it.

    `write_experiment_git_record` costs time proportional to `files_changed`.
    Stamping after it would let a branch that finished first but wrote a big
    record lose a tie-break to one that finished later and wrote less — the
    ranking would partly measure record size.
    """
    stub_run(execution_id="E-1", status="succeeded")
    entered: list[datetime] = []

    def _slow_record(root: Path, payload: dict[str, Any]) -> Path:
        del payload
        entered.append(datetime.now(UTC))
        time.sleep(0.05)
        return root / "record.json"

    monkeypatch.setattr(experiment_mod, "write_experiment_git_record", _slow_record)

    events = _run(tmp_path)

    stamped = datetime.fromisoformat(events[0][1]["completed_at"])
    assert stamped <= entered[0], "the stamp was taken after the record write began"


def test_stamps_order_by_finish_so_the_earliest_branch_can_win(
    tmp_path: Path, stub_run: Callable[..., None]
) -> None:
    """Promotion breaks a metric tie by earliest finisher; that needs ordering.

    A stamp truncated to whole seconds would compare equal for two branches
    finishing in the same second and leave the tie-break arbitrary again.
    """
    stub_run(execution_id="E-1", status="succeeded")

    first = _run(tmp_path / "a")[0][1]["completed_at"]
    second = _run(tmp_path / "b")[0][1]["completed_at"]

    assert datetime.fromisoformat(first) < datetime.fromisoformat(second)
    # Lexicographic order agrees with chronological order for these stamps —
    # fixed-width UTC sharing one offset. Worth pinning because sorting the
    # raw strings is a reasonable thing for a consumer to do, and a local-time
    # or offset-varying stamp would break it silently.
    assert first < second
