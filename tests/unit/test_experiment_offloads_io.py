"""M11: experiment bookkeeping must not run on the event loop.

Under K-way fan-out every branch shares one loop, so a branch's metrics read
and record write would stall its siblings. Rationale: design doc §8.
"""

from __future__ import annotations

import threading
import time
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

from labpilot.research_engine.agents import experiment as experiment_mod
from labpilot.research_engine.agents.experiment import ExperimentSpecialist


def test_the_metrics_read_and_record_write_leave_the_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are dispatched to a worker rather than run inline."""
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ran_on: dict[str, str] = {}
    real_load = experiment_mod._load_metrics
    real_write = experiment_mod.write_experiment_git_record

    def _load(root: Path) -> dict[str, Any]:
        ran_on["load_metrics"] = threading.current_thread().name
        return real_load(root)

    def _write(root: Path, payload: dict[str, Any]) -> Path:
        ran_on["write_record"] = threading.current_thread().name
        return real_write(root, payload)

    monkeypatch.setattr(experiment_mod, "_load_metrics", _load)
    monkeypatch.setattr(experiment_mod, "write_experiment_git_record", _write)

    anyio.run(
        lambda: ExperimentSpecialist().execute(
            training_task(), experiment_workspace(tmp_path), bundle()
        )
    )

    loop_thread = threading.main_thread().name
    assert set(ran_on) == {"load_metrics", "write_record"}
    assert ran_on["load_metrics"] != loop_thread
    assert ran_on["write_record"] != loop_thread


def test_the_metrics_stat_leaves_the_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `is_file` deciding the metrics ArtifactRef is the third blocking call."""
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = experiment_workspace(tmp_path)
    metrics_path = ws.root / "metrics.json"
    metrics_path.write_text('{"rmse": 1.0}', encoding="utf-8")

    real_is_file = Path.is_file
    stat_threads: list[str] = []

    def _is_file(self: Path) -> bool:
        # Matched on the exact path, not the filename: a stat of some other
        # `metrics.json` must not be mistaken for the call under test.
        if self == metrics_path:
            stat_threads.append(threading.current_thread().name)
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", _is_file)

    anyio.run(lambda: ExperimentSpecialist().execute(training_task(), ws, bundle()))

    assert stat_threads, "the metrics path was never stat'd"
    assert threading.main_thread().name not in stat_threads


def test_the_offloaded_calls_still_read_and_write_the_right_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where a call runs says nothing about whether it still works.

    `run_sync` takes the argument separately from the callable, so a wrong
    path stays off the loop and still returns — silently emptying `metrics`
    for every run. The thread assertions above cannot see that.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ws = experiment_workspace(tmp_path)
    (ws.root / "metrics.json").write_text('{"rmse": 1.5}', encoding="utf-8")
    seen: list[tuple[str, dict[str, Any]]] = []

    agent = ExperimentSpecialist(on_event=lambda e, p: seen.append((e, p)))
    anyio.run(lambda: agent.execute(training_task(), ws, bundle()))

    ((_, payload),) = seen
    assert payload["metrics"] == {"rmse": 1.5}
    refs = {r["kind"]: r for r in payload["refs"]}
    assert "metrics" in refs
    record = Path(refs["experiment"]["path"])
    # Under the workspace root, not merely somewhere: handing the write a
    # different directory still produces a file that exists.
    assert record.is_file()
    assert record.is_relative_to(ws.root)


def test_a_slow_record_write_does_not_stall_the_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property fan-out depends on: one branch's I/O yields to its siblings.

    Ticks are counted across the blocking window only, so a stall reads as
    zero rather than being masked by ticks from the rest of `execute`.
    """
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ticks = 0
    window: list[int] = []
    real_write = experiment_mod.write_experiment_git_record

    def _slow_write(root: Path, payload: dict[str, Any]) -> Path:
        window.append(ticks)
        time.sleep(0.15)
        window.append(ticks)
        return real_write(root, payload)

    monkeypatch.setattr(experiment_mod, "write_experiment_git_record", _slow_write)

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await anyio.sleep(0.005)
            ticks += 1

    async def _main() -> None:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_ticker)
            await ExperimentSpecialist().execute(
                training_task(), experiment_workspace(tmp_path), bundle()
            )
            tg.cancel_scope.cancel()

    anyio.run(_main)

    assert len(window) == 2, "the slow write never ran"
    # ~30 ticks fit in 0.15s; 0 if the write held the loop.
    assert window[1] - window[0] >= 5
