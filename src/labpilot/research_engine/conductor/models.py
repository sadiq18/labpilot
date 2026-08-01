"""Conductor domain models — sessions, OS tasks, decisions, approvals."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SessionStatus = Literal[
    "running", "paused", "sleeping", "waiting", "failed", "retry", "completed"
]
TaskStatus = Literal[
    "pending", "running", "completed", "failed", "retry", "blocked", "cancelled"
]
ApprovalDecision = Literal["approve", "reject"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Objective(BaseModel):
    """Research objective attached to a conduct session."""

    goal: str
    target_metric: str | None = None
    target_value: float | None = None
    max_steps: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConductSession(BaseModel):
    """Durable Conductor session for one competition + goal."""

    id: str
    competition: str
    goal: str
    status: SessionStatus = "running"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    @property
    def objective(self) -> Objective:
        return Objective(
            goal=self.goal,
            target_metric=self.metadata.get("target_metric"),
            target_value=self.metadata.get("target_value"),
            max_steps=self.metadata.get("max_steps"),
            metadata={
                k: v
                for k, v in self.metadata.items()
                if k not in {"target_metric", "target_value", "max_steps"}
            },
        )


class OsTask(BaseModel):
    """One OS-level work item: invoke a single registered tool."""

    id: str
    session_id: str
    tool_name: str
    status: TaskStatus = "pending"
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 1
    args: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    decision_id: str | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None


class ApprovalResult(BaseModel):
    """Operator gate outcome for a gated tool."""

    decision: ApprovalDecision
    comment: str = ""
    gated_tool: str
    at: str = Field(default_factory=_now)
    decision_id: str | None = None
    task_id: str | None = None


class NextAction(BaseModel):
    """Structured policy output — must name an allowlisted tool or stop."""

    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    stop: bool = False


class DecisionRecord(BaseModel):
    """Append-only decision / event log row."""

    id: str
    session_id: str
    tool_name: str | None = None
    rationale: str = ""
    stop: bool = False
    args: dict[str, Any] = Field(default_factory=dict)
    observe: dict[str, Any] = Field(default_factory=dict)
    approval: ApprovalResult | None = None
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    task_id: str | None = None
    created_at: str = Field(default_factory=_now)


class OperatorFeedback(BaseModel):
    """Durable operator comment from an approval gate."""

    id: str
    session_id: str
    gated_tool: str
    decision: ApprovalDecision
    comment: str = ""
    decision_id: str | None = None
    task_id: str | None = None
    created_at: str = Field(default_factory=_now)
