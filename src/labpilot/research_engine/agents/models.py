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
