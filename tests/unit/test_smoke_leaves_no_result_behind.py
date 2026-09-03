"""The smoke gate proves a script runs; it must not leave a score behind.

The gate executes the real `train.py` from the workspace root, so anything the
script writes there lands in the files a training run writes. `LABPILOT_SMOKE`
asks it to take a short path — a slice of the rows, a couple of folds — and the
result is a plausible number in the file a trained result belongs in.

Nothing downstream separates them on its own. `_metrics_written_since` only asks
whether the mtime beats the step's start, and the gate runs inside that step
(plan order is smoke → train under one `run_experiment`), so a training run that
then fails inherits the gate's numbers and reports success. That is the failure
`PLACEHOLDER_STATUSES` was written for, arriving by a path it does not cover.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labpilot.research_engine.evidence.builder import (
    PLACEHOLDER_STATUSES,
    is_placeholder_metrics,
)
from labpilot.research_engine.execution.capabilities.verification.capability import (
    _SMOKE_MUST_NOT_KEEP,
    _artifacts_preserved,
)


def test_a_score_written_during_smoke_does_not_survive(tmp_path: Path) -> None:
    """The whole point: what the gate's subprocess writes is not kept."""
    with _artifacts_preserved(tmp_path, _SMOKE_MUST_NOT_KEEP):
        (tmp_path / "metrics.json").write_text('{"cv_roc_auc": 0.81}', encoding="utf-8")
        (tmp_path / "submission.csv").write_text("id,target\n1,0.5\n", encoding="utf-8")

    assert not (tmp_path / "metrics.json").exists()
    assert not (tmp_path / "submission.csv").exists()


def test_an_earlier_result_is_returned_untouched(tmp_path: Path) -> None:
    """A previous run's artifacts must come back byte for byte **and** with
    their original mtime: the freshness check downstream reads mtime alone, so
    restoring the right bytes under a new timestamp would still read as "this
    run produced it" — the exact confusion being removed."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"cv_roc_auc": 0.957}', encoding="utf-8")
    import os

    os.utime(metrics, (1_600_000_000, 1_600_000_000))
    before = metrics.stat().st_mtime

    with _artifacts_preserved(tmp_path, _SMOKE_MUST_NOT_KEEP):
        metrics.write_text('{"cv_roc_auc": 0.42}', encoding="utf-8")

    assert json.loads(metrics.read_text(encoding="utf-8")) == {"cv_roc_auc": 0.957}
    assert metrics.stat().st_mtime == before


def test_artifacts_are_restored_even_when_the_run_raises(tmp_path: Path) -> None:
    """The gate returns early on `TimeoutExpired`, which is the case that matters
    most — a script killed at 120s is the one most likely to have written a
    partial file."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"cv_roc_auc": 0.957}', encoding="utf-8")

    with pytest.raises(TimeoutError):  # noqa: PT012 — the write is the point
        with _artifacts_preserved(tmp_path, _SMOKE_MUST_NOT_KEEP):
            metrics.write_text("partial", encoding="utf-8")
            raise TimeoutError

    assert json.loads(metrics.read_text(encoding="utf-8")) == {"cv_roc_auc": 0.957}


def test_a_smoke_status_is_read_as_a_placeholder() -> None:
    """The second lock, for a `metrics.json` that reaches a reader some other
    way — a run started by hand, a gate that died before restoring."""
    assert "smoke" in PLACEHOLDER_STATUSES
    assert is_placeholder_metrics({"cv_roc_auc": 0.81, "status": "smoke"})
    assert not is_placeholder_metrics({"cv_roc_auc": 0.81})
