"""Read/write adapters for workspace submission result JSON."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.execution.outcome import (
    submission_csv_path,
    submission_result_path,
)

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["submission"]


class SubmissionRecord(BaseModel):
    """Typed view of a submission result blob and optional CSV path.

    Maps onto ``artifacts/submission_result_<execution_id>.json`` in a workspace.
    """

    schema_id: str = SCHEMA_ID
    competition: str = ""
    execution_id: str
    csv_path: str | None = None
    result_path: str | None = None
    public_score: float | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def write_submission_record(
    workspace_root: Path,
    execution_id: str,
    payload: dict[str, Any],
    *,
    competition: str = "",
    produced_by: str = "submit",
) -> tuple[SubmissionRecord, ArtifactRef]:
    """Write submission result JSON and return the record with an :class:`ArtifactRef`."""
    root = Path(workspace_root)
    result_path = submission_result_path(root, execution_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv = submission_csv_path(root, execution_id)
    record = SubmissionRecord(
        competition=competition,
        execution_id=execution_id,
        csv_path=str(csv) if csv.is_file() else None,
        result_path=str(result_path),
        public_score=_as_float(payload.get("public_score")),
        status=str(payload.get("status")) if payload.get("status") is not None else None,
        payload=payload,
    )
    _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
    ref = ArtifactRef(
        kind="submission",
        id=f"submission:{execution_id}",
        schema_id=SCHEMA_ID,
        path=str(result_path),
        competition=competition or None,
    )
    return record, ref


def read_submission_record(
    workspace_root: Path,
    execution_id: str,
    *,
    competition: str = "",
) -> SubmissionRecord | None:
    """Load a submission record from the workspace, or ``None`` if missing/invalid."""
    root = Path(workspace_root)
    result_path = submission_result_path(root, execution_id)
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    csv = submission_csv_path(root, execution_id)
    return SubmissionRecord(
        competition=competition,
        execution_id=execution_id,
        csv_path=str(csv) if csv.is_file() else None,
        result_path=str(result_path),
        public_score=_as_float(payload.get("public_score")),
        status=str(payload.get("status")) if payload.get("status") is not None else None,
        payload=payload,
    )


def _as_float(value: Any) -> float | None:
    """Coerce a payload value to float, or ``None`` if unset/invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
