"""Training must write metrics, not merely find some.

Measured on rogii 2026-08-09: execution E-227 reported **succeeded** and plan
P-025 went **done** against a `metrics.json` written the previous evening. A
green plan with no result, and the number on the card belonged to a different
experiment.

`run_experiment` closed this hole with `_metrics_written_since`. The Engineer
path — `research run`, which is what a plan actually goes through — never had
the guard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


class _Result:
    returncode = 0
    stdout = ""
    stderr = ""


class _Runner:
    """A training run that exits 0 and writes nothing."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, timeout=None):
        return _Result()

    def save_run_log(self, result):
        path = self.root / "artifacts" / "run.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    def collect_artifacts(self):
        return {}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr(
        "labpilot.research_engine.execution.training.runner.TrainingRunner", _Runner
    )
    return tmp_path


def _run(workspace):
    from labpilot.research_engine.execution.capabilities.training.capability import (
        TrainingCapability,
    )

    class _Ctx:
        workspace_root = workspace
        constraints: dict = {}

        class task:
            id = "T-1"
            metadata: dict = {}

        class plan:
            id = "P-1"

        class execution:
            id = "E-1"

    return TrainingCapability().execute(_Ctx())


def test_a_stale_metrics_file_does_not_count_as_a_result(workspace):
    """The exact E-227 shape: yesterday's metrics, today's green plan."""
    stale = workspace / "metrics.json"
    stale.write_text(json.dumps({"cv_rmse": 194.8}), encoding="utf-8")
    old = time.time() - 86_400
    import os

    os.utime(stale, (old, old))

    result = _run(workspace)

    assert result.passed is False
    assert "predates this run" in (result.error or "")


def test_no_metrics_at_all_still_fails_with_its_own_message(workspace):
    """The two causes need different messages — a missing file points at the
    __main__ guard, a stale one at the previous execution."""
    result = _run(workspace)

    assert result.passed is False
    assert "__main__" in (result.error or "")


def test_metrics_written_by_this_run_pass(workspace):
    """The carve-out must not cost the behaviour it guards."""

    class _WritingRunner(_Runner):
        def run(self, timeout=None):
            (self.root / "metrics.json").write_text(json.dumps({"cv_rmse": 1.0}), encoding="utf-8")
            return _Result()

    import labpilot.research_engine.execution.training.runner as runner_mod

    runner_mod.TrainingRunner = _WritingRunner

    result = _run(workspace)

    assert result.passed is True, result.error
    assert result.metrics["cv_rmse"] == 1.0
