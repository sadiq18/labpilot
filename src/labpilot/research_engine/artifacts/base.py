"""Shared identity and attribution types for artifact adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# Stable ids so artifact formats can version independently of model classes.
ARTIFACT_SCHEMA_IDS: dict[str, str] = {
    "competition_analysis": "labpilot.artifact.competition_analysis/v1",
    "research_plan": "labpilot.artifact.research_plan/v1",
    "execution": "labpilot.artifact.execution/v1",
    "evidence_card": "labpilot.artifact.evidence_card/v1",
    "reflection": "labpilot.artifact.reflection/v1",
    "submission": "labpilot.artifact.submission/v1",
}


class ArtifactRef(BaseModel):
    """Location handle for a persisted artifact.

    Identifies *what* was written (``kind`` / ``id`` / ``schema_id``) and optionally
    *where* (filesystem ``path``) under a competition.
    """

    kind: str
    id: str
    schema_id: str
    path: str | None = None
    competition: str | None = None


class ArtifactMeta(BaseModel):
    """Provenance recorded alongside an artifact write."""

    schema_id: str
    produced_by: str = ""
    produced_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    decision_id: str | None = None
    task_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
