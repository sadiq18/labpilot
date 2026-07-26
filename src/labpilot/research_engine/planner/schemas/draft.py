"""Slim LLM draft shapes — lowered into ``ResearchPlan`` by the compiler.

The Planning Engine returns this lightweight payload only. IDs, timestamps,
verification defaults, and status belong in deterministic Python lowering, not
in the LLM response.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.research_engine.planner.schemas.task_types import TaskType


class DraftTask(BaseModel):
    """One DAG node keyed by a local ``key``; ``depends_on`` references keys."""

    key: str
    type: TaskType
    description: str = ""
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class ResearchPlanDraft(BaseModel):
    """Typed Planning Engine output — no ids, no timestamps, no verification."""

    goal: str = ""
    current_state: str = ""
    expected_outcome: str = ""
    risk: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    rollback: str = ""
    artifacts: list[str] = Field(default_factory=list)
    tasks: list[DraftTask] = Field(default_factory=list)
