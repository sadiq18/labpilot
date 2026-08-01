"""Tool descriptors and invocation results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.artifacts.base import ArtifactRef


class ToolResult(BaseModel):
    """Outcome of a tool invocation — artifact refs plus optional payload."""

    refs: list[ArtifactRef] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


ToolHandler = Callable[..., ToolResult]


class ToolDescriptor(BaseModel):
    """Named tool with typed I/O expectations and an in-process handler."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[str] = Field(default_factory=list)
    required_workspace_fields: list[str] = Field(
        default_factory=lambda: ["competition", "knowledge_dir", "root"]
    )
    handler: ToolHandler

    model_config = {"arbitrary_types_allowed": True}
