import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExperimentRecord(BaseModel):
    run_id: str
    competition: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


class ExperimentStore:
    """Local JSON-based experiment tracking for P0."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.experiment_dir = run_dir / "experiment"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ExperimentRecord) -> Path:
        path = self.experiment_dir / "record.json"
        path.write_text(record.model_dump_json(indent=2))
        return path

    def load(self) -> ExperimentRecord | None:
        path = self.experiment_dir / "record.json"
        if not path.exists():
            return None
        return ExperimentRecord.model_validate(json.loads(path.read_text()))
