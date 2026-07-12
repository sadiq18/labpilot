import json
from pathlib import Path

import pytest

from labpilot.tracking.index import RunDiff, diff_runs
from labpilot.tracking.store import ExperimentRecord, ExperimentStore


def _seed_run(
    runs_dir: Path,
    run_id: str,
    *,
    metrics: dict[str, float],
    params: dict,
    metadata: dict | None = None,
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "competition": "titanic",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "status": "completed",
        "stages": [],
        "metadata": metadata or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    ExperimentStore(run_dir).save(
        ExperimentRecord(run_id=run_id, competition="titanic", metrics=metrics, params=params)
    )


def test_diff_runs_reports_metric_and_param_deltas(tmp_path: Path):
    runs_dir = tmp_path / "runs"
    _seed_run(
        runs_dir,
        "base-run",
        metrics={"cv_accuracy": 0.75},
        params={"model_params": {"learning_rate": 0.05}, "template": "tabular_classification"},
    )
    _seed_run(
        runs_dir,
        "child-run",
        metrics={"cv_accuracy": 0.80},
        params={"model_params": {"learning_rate": 0.1}, "template": "tabular_classification"},
        metadata={"parent_run_id": "base-run", "iteration": 1, "improvement_strategy": "tune"},
    )

    result = diff_runs(runs_dir, "base-run", "child-run")
    assert isinstance(result, RunDiff)
    assert result.metric_deltas["cv_accuracy"] == pytest.approx(0.05)
    assert "model_params" in result.param_changes
