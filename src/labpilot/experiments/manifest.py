"""Legacy ``runs/<run_id>/manifest.json`` helpers (inspect / experiment graph).

Not the Research Engineer orchestrator — that lives under
``research_engine.execution``. Kept here because experiment assembly and
CLI status/report still read historical run manifests.
"""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    # RunManifest-level only (never set on an individual StageRecord): all
    # requested stages finished without error, but they didn't reach the end
    # of a historical linear pipeline run (legacy runs/ artifacts).
    PARTIAL = "partial"


class StageRecord(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str
    competition: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: StageStatus = StageStatus.PENDING
    stages: list[StageRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def stage(self, name: str) -> StageRecord | None:
        for record in self.stages:
            if record.name == name:
                return record
        return None

    def mark_running(self, name: str) -> None:
        record = self._get_or_create(name)
        record.status = StageStatus.RUNNING
        record.started_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_completed(self, name: str, artifacts: list[str] | None = None) -> None:
        record = self._get_or_create(name)
        record.status = StageStatus.COMPLETED
        record.finished_at = datetime.now()
        if artifacts:
            record.artifacts = artifacts
        self.updated_at = datetime.now()

    def mark_failed(self, name: str, error: str) -> None:
        record = self._get_or_create(name)
        record.status = StageStatus.FAILED
        record.finished_at = datetime.now()
        record.error = error
        self.status = StageStatus.FAILED
        self.updated_at = datetime.now()

    def mark_skipped(self, name: str, artifacts: list[str] | None = None) -> None:
        record = self._get_or_create(name)
        record.status = StageStatus.SKIPPED
        record.finished_at = datetime.now()
        if artifacts:
            record.artifacts = artifacts
        self.updated_at = datetime.now()

    def _get_or_create(self, name: str) -> StageRecord:
        record = self.stage(name)
        if record is None:
            record = StageRecord(name=name)
            self.stages.append(record)
        return record


def generate_run_id(competition: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = competition.replace("/", "-").lower()
    return f"{timestamp}-{slug}"


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def load_manifest(run_dir: Path) -> RunManifest:
    path = manifest_path(run_dir)
    return RunManifest.model_validate_json(path.read_text())


def save_manifest(run_dir: Path, manifest: RunManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(run_dir).write_text(manifest.model_dump_json(indent=2))


def get_run_dir(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / run_id


def find_manifest(runs_dir: Path | object, run_id: str) -> RunManifest:
    """Load a run manifest.

    ``runs_dir`` may be a ``Path`` or an object with a ``runs_dir`` attribute
    (e.g. ``AppConfig``) for backward compatibility with inspect CLI helpers.
    """
    base = runs_dir if isinstance(runs_dir, Path) else Path(getattr(runs_dir, "runs_dir"))
    run_dir = get_run_dir(base, run_id)
    if not (run_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return load_manifest(run_dir)
