"""M11: experiment bookkeeping must not run on the event loop.

Under K-way fan-out every branch shares one loop, so a branch's metrics read
and record write would stall its siblings. Rationale: design doc §8.
"""

from __future__ import annotations

import os
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


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (None, ({}, False)),
        ("{not json", ({}, True)),
        ('{"rmse": 1.5}', ({"rmse": 1.5}, True)),
        ("[1, 2]", ({"value": [1, 2]}, True)),
    ],
    ids=["absent", "corrupt", "valid-dict", "valid-non-dict"],
)
def test_load_metrics_reports_contents_and_existence(
    tmp_path: Path, contents: str | None, expected: tuple[dict[str, Any], bool]
) -> None:
    """Existence is answered without a stat when the read already proves it.

    The corrupt case is the one that needs the stat: no metrics, but the file
    is still an artifact worth pointing at.
    """
    if contents is not None:
        (tmp_path / "metrics.json").write_text(contents, encoding="utf-8")

    assert experiment_mod._load_metrics(tmp_path) == expected


def test_load_metrics_treats_a_directory_as_absent(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").mkdir()

    assert experiment_mod._load_metrics(tmp_path) == ({}, False)


def test_load_metrics_survives_a_file_that_is_not_utf8(tmp_path: Path) -> None:
    """`read_text` raises UnicodeDecodeError, which is a ValueError not an OSError.

    Every other malformed case degrades to no metrics; this one used to
    propagate and take a finished experiment down with it.
    """
    (tmp_path / "metrics.json").write_bytes(b"\x80\x81\x82 not utf-8")

    assert experiment_mod._load_metrics(tmp_path) == ({}, True)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads regardless of mode")
def test_load_metrics_still_references_an_unreadable_file(tmp_path: Path) -> None:
    """Present but unreadable: no metrics, yet still an artifact to point at.

    The only case where the `OSError` arm must answer True, so it is what
    stops that arm being "simplified" to `return {}, False` on the reasoning
    that an OSError means the file is not there. Absent and directory both
    take the same arm and want False.
    """
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"rmse": 1.5}', encoding="utf-8")
    metrics.chmod(0o000)
    try:
        assert experiment_mod._load_metrics(tmp_path) == ({}, True)
    finally:
        metrics.chmod(0o644)


def test_the_metrics_read_and_record_write_leave_the_loop_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both are dispatched to a worker rather than run inline."""
    stub_experiment_io(monkeypatch, execution_id="E-1", status="succeeded")
    ran_on: dict[str, str] = {}
    real_load = experiment_mod._load_metrics
    real_write = experiment_mod.write_experiment_git_record

    def _load(root: Path) -> tuple[dict[str, Any], bool]:
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
