"""Specialist task and registry descriptor models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.agents.ports import Agent


class AgentTask(BaseModel):
    """Minimal specialist work item (Conductor maps OsTask → this later)."""

    id: str
    capability: str = "implement"
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def as_agent_task(task: object) -> AgentTask:
    """Normalize OsTask / dict / AgentTask into :class:`AgentTask`."""
    if isinstance(task, AgentTask):
        return task
    if isinstance(task, dict):
        return AgentTask.model_validate(task)
    task_id = str(getattr(task, "id", None) or getattr(task, "task_id", None) or "T-agent")
    capability = str(
        getattr(task, "capability", None)
        or getattr(task, "tool_name", None)
        or "implement"
    )
    description = str(getattr(task, "description", None) or getattr(task, "goal", None) or "")
    metadata = getattr(task, "metadata", None) or getattr(task, "args", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return AgentTask(
        id=task_id,
        capability=capability,
        description=description,
        metadata=dict(metadata),
    )


class SpecialistDescriptor(BaseModel):
    """Advertisement for registry routing — capability + cost/duration hints."""

    name: str
    capabilities: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    cost_hint: float | None = None
    duration_hint: float | None = None
    agent: Agent

    model_config = {"arbitrary_types_allowed": True}
