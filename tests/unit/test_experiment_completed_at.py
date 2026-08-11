"""M11: `ExperimentCompleted` carries when the run finished.

The promotion subscriber ranks branches on the metric and breaks a tie by
earliest finisher, so the payload carries a finish time. The stamp is only
meaningful on a run that actually finished: the failure path shares this
dict, and a
`completed_at` on a run that died would assert the completion
`test_failed_run_is_not_completed.py` exists to deny.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
from helpers.experiment_harness import (
    bundle,
    experiment_workspace,
    stub_git_snapshot,
    stub_run_plan,
    training_task,
)

from labpilot.research_engine.agents import experiment as experiment_mod
from labpilot.research_engine.agents.events import (
    EXPERIMENT_COMPLETED,
    MODEL_FAILED,
    EventBus,
)
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.subscribers import install_evidence_refresh_subscriber


@pytest.fixture
def stub_run(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Returns a setter for the outcome `run_plan` reports.

    Installed once up front too, so a test that takes this fixture and forgets
    to call the setter still cannot reach the real plan runner.
    """

    stub_git_snapshot(monkeypatch)

    def _set(**data: Any) -> None:
        stub_run_plan(monkeypatch, **data)

    _set()
    return _set


def _run(tmp_path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Execute the specialist, returning every event it emitted."""
    seen: list[tuple[str, dict[str, Any]]] = []
    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))

    anyio.run(lambda: agent.execute(training_task(), experiment_workspace(tmp_path), bundle()))
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
    entered: dict[str, datetime] = {}

    def _slow_metrics(root: Path) -> tuple[dict[str, Any], bool]:
        del root
        entered["metrics"] = datetime.now(UTC)
        time.sleep(0.05)
        return {}, False

    def _slow_record(root: Path, payload: dict[str, Any]) -> Path:
        del payload
        entered["record"] = datetime.now(UTC)
        time.sleep(0.05)
        return root / "record.json"

    # Both halves of the block the comment names, not just the record write:
    # with only the latter pinned, the stamp could slide past the metrics read
    # and the suite would stay green while the comment became false.
    monkeypatch.setattr(experiment_mod, "_load_metrics", _slow_metrics)
    monkeypatch.setattr(experiment_mod, "write_experiment_git_record", _slow_record)

    events = _run(tmp_path)

    stamped = datetime.fromisoformat(events[0][1]["completed_at"])
    assert stamped <= entered["metrics"], "the stamp was taken after the metrics read began"
    assert stamped <= entered["record"], "the stamp was taken after the record write began"


def test_the_live_subscriber_tolerates_the_new_key(tmp_path: Path) -> None:
    """The key has to survive the path it is actually published into.

    Every other test here reads the payload straight off the emitter. This
    puts one through `install_evidence_refresh_subscriber` on a real
    `EventBus`, so the addition is checked against the production dispatch
    path and a real reader rather than assumed inert because dicts ignore
    extra keys.
    """
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    install_evidence_refresh_subscriber(bus)

    bus.publish(
        EXPERIMENT_COMPLETED,
        {
            "experiment_id": "exp_E-1",
            "execution_id": "E-1",
            "plan_id": "P-001",
            "competition": "demo",
            "workspace_root": str(tmp_path),
            "metrics": {"rmse": 1.0},
            "status": "succeeded",
            "completed_at": datetime.now(UTC).isoformat(),
        },
    )

    note = tmp_path / "artifacts" / "evidence_refresh_demo.json"
    assert note.is_file(), "the subscriber refused a payload carrying completed_at"
    written = json.loads(note.read_text(encoding="utf-8"))
    assert written["execution_id"] == "E-1"
    # The stamp does *not* reach the note: the subscriber builds its own dict
    # from named keys. Recorded so promotion reads `completed_at` off the
    # event rather than discovering by a silent miss that the artifact has none.
    assert "completed_at" not in written, (
        "the evidence note now carries completed_at — if persisting the stamp "
        "was a deliberate decision, update this assertion; it pins current "
        "behaviour, not a requirement (design doc §8, tie-break)"
    )


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
