import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.orchestrator.manifest import load_manifest
from labpilot.tracking.store import ExperimentRecord, ExperimentStore


class RunIndexEntry(BaseModel):
    run_id: str
    competition: str
    status: str
    parent_run_id: str | None = None
    iteration: int = 0
    improvement_strategy: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class RunDiff(BaseModel):
    base_run_id: str
    compare_run_id: str
    base_metrics: dict[str, float] = Field(default_factory=dict)
    compare_metrics: dict[str, float] = Field(default_factory=dict)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    base_params: dict[str, Any] = Field(default_factory=dict)
    compare_params: dict[str, Any] = Field(default_factory=dict)
    param_changes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    lineage: dict[str, Any] = Field(default_factory=dict)
    submission_notes: dict[str, str] = Field(default_factory=dict)


def scan_runs(runs_dir: Path) -> list[RunIndexEntry]:
    if not runs_dir.is_dir():
        return []

    entries: list[RunIndexEntry] = []
    for run_dir in sorted(runs_dir.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = load_manifest(run_dir)
        record = ExperimentStore(run_dir).load()
        metrics = record.metrics if record else {}
        params = record.params if record else {}
        entries.append(
            RunIndexEntry(
                run_id=manifest.run_id,
                competition=manifest.competition,
                status=manifest.status.value,
                parent_run_id=manifest.metadata.get("parent_run_id"),
                iteration=int(manifest.metadata.get("iteration", 0)),
                improvement_strategy=manifest.metadata.get("improvement_strategy"),
                metrics=metrics,
                params=params,
            )
        )
    return entries


def diff_runs(runs_dir: Path, base_run_id: str, compare_run_id: str) -> RunDiff:
    base_dir = runs_dir / base_run_id
    compare_dir = runs_dir / compare_run_id
    if not (base_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Base run not found: {base_run_id}")
    if not (compare_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"Compare run not found: {compare_run_id}")

    base_manifest = load_manifest(base_dir)
    compare_manifest = load_manifest(compare_dir)
    base_record = ExperimentStore(base_dir).load() or ExperimentRecord(
        run_id=base_run_id, competition=base_manifest.competition
    )
    compare_record = ExperimentStore(compare_dir).load() or ExperimentRecord(
        run_id=compare_run_id, competition=compare_manifest.competition
    )

    metric_deltas: dict[str, float] = {}
    all_metric_keys = set(base_record.metrics) | set(compare_record.metrics)
    for key in sorted(all_metric_keys):
        base_value = base_record.metrics.get(key)
        compare_value = compare_record.metrics.get(key)
        if isinstance(base_value, (int, float)) and isinstance(compare_value, (int, float)):
            metric_deltas[key] = float(compare_value) - float(base_value)

    param_changes: dict[str, dict[str, Any]] = {}
    all_param_keys = set(base_record.params) | set(compare_record.params)
    for key in sorted(all_param_keys):
        base_value = base_record.params.get(key)
        compare_value = compare_record.params.get(key)
        if base_value != compare_value:
            param_changes[key] = {"base": base_value, "compare": compare_value}

    lineage = {
        "base_iteration": int(base_manifest.metadata.get("iteration", 0)),
        "compare_iteration": int(compare_manifest.metadata.get("iteration", 0)),
        "compare_parent_run_id": compare_manifest.metadata.get("parent_run_id"),
        "compare_strategy": compare_manifest.metadata.get("improvement_strategy"),
    }

    submission_notes = {
        "base": _submission_summary(base_dir),
        "compare": _submission_summary(compare_dir),
    }

    return RunDiff(
        base_run_id=base_run_id,
        compare_run_id=compare_run_id,
        base_metrics=base_record.metrics,
        compare_metrics=compare_record.metrics,
        metric_deltas=metric_deltas,
        base_params=base_record.params,
        compare_params=compare_record.params,
        param_changes=param_changes,
        lineage=lineage,
        submission_notes=submission_notes,
    )


def _submission_summary(run_dir: Path) -> str:
    path = run_dir / "submission_result.json"
    if not path.is_file():
        return "not submitted"
    data = json.loads(path.read_text())
    status = data.get("status", "unknown")
    public_score = data.get("public_score")
    if public_score is not None:
        return f"{status} (public score: {public_score})"
    return str(status)
