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
