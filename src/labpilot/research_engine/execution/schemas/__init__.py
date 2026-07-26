"""Execution attempt and task evidence models (Research Engineer)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ExecutionStatus = Literal[
    "pending", "running", "succeeded", "failed", "cancelled"
]


class TaskEvidence(BaseModel):
    """Structured evidence for one task attempt."""

    task_id: str
    execution_id: str
    capability: str = ""
    passed: bool = True
    summary: str = ""
    checks: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ResearchExecution(BaseModel):
    """One attempt to run a research plan (``E-xxx``)."""

    id: str
    plan_id: str
    competition: str = ""
    status: ExecutionStatus = "pending"
    workspace_path: str | None = None
    runtime_target: str | None = None
    experiment_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
