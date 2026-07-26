"""Pydantic shapes for the durable plan/task model.

The DB (``research_plans`` / ``research_tasks`` / ``research_task_deps``) is the
source of record; these models are the typed contract the compiler emits and the
PlanStore round-trips. ``parent_task_id`` groups/sequences tasks, but
``dependencies`` carries the true DAG edges (parallel branches).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.planner.schemas.task_types import (
    PlanStatus,
    RuntimeTarget,
    TaskStatus,
    TaskType,
)


class TaskVerification(BaseModel):
    """What "success" means for a task, and how to recover from failure."""

    expected_output: str = ""
    check: str = ""
    failure_recovery: str = ""


class RetryPolicy(BaseModel):
    max_retries: int = 0
    abort_on_failure: bool = True


class ResearchTask(BaseModel):
    id: str
    plan_id: str
    parent_task_id: str | None = None
    type: TaskType
    description: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    #: Task ids this task depends on; persisted to ``research_task_deps``.
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    order: int = 0
    verification: TaskVerification = Field(default_factory=TaskVerification)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    estimated_cost: float | None = None
    estimated_time: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    id: str
    competition: str
    hypothesis_id: str
    goal: str = ""
    current_state: str = ""
    expected_outcome: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    priority: int = 0
    estimated_gain: float = 0.0
    risk: str = ""
    estimated_cost: float | None = None
    estimated_duration: str | None = None
    runtime_target: RuntimeTarget | None = None
    tasks: list[ResearchTask] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    rollback: str = ""
    generated_by: Literal["llm", "rule_engine"] = "rule_engine"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    notes: list[str] = Field(default_factory=list)
