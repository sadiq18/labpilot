"""Tool descriptors and invocation results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from labpilot.research_engine.artifacts.base import ArtifactRef


class ToolResult(BaseModel):
    """Outcome of a tool invocation — artifact refs plus optional payload."""

    refs: list[ArtifactRef] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


ToolHandler = Callable[..., ToolResult]


class ToolDescriptor(BaseModel):
    """Named tool with typed I/O expectations and an in-process handler.

    ``capability_status`` and ``varies_by`` exist so a named tool cannot
    silently outrun what it actually does (M15 —
    docs/research-os/design/12-capability-audit.md). A tool that cannot
    produce a different outcome given different inputs is not a capability,
    it is a fixed step, and the catalog should say so.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[str] = Field(default_factory=list)
    required_workspace_fields: list[str] = Field(
        default_factory=lambda: ["competition", "knowledge_dir", "root"]
    )
    handler: ToolHandler
    #: What this tool can vary. ``[]`` is the *correct* value for a
    #: ``"fixed"`` tool, not a placeholder for "not filled in yet" — see the
    #: ``capability_status`` validator below for what actually is required.
    varies_by: list[str] = Field(default_factory=list)
    #: No default, deliberately. A default value is never "missing" under
    #: Pydantic, so a tool added without deciding its status would silently
    #: pass as whatever the default was rather than failing to construct.
    #: Required forces every descriptor — catalog or test double — to state
    #: this explicitly.
    capability_status: Literal["real", "partial", "fixed"]

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _status_and_variance_agree(self) -> ToolDescriptor:
        if self.capability_status == "real" and not self.varies_by:
            raise ValueError(
                f"{self.name}: capability_status='real' but varies_by=[] — "
                "declare what input changes the output, or downgrade the status"
            )
        if self.capability_status == "fixed" and self.varies_by:
            # The two halves contradict each other: "same output whatever the
            # input" and "these inputs change the output". Left unchecked, the
            # contract harness would branch on `varies_by` and try to prove
            # variance for a tool the catalog calls fixed, and
            # `research tools list` would print `fixed` beside a varies-by
            # column — an operator cannot act on that.
            raise ValueError(
                f"{self.name}: capability_status='fixed' but varies_by="
                f"{self.varies_by} — a fixed step varies by nothing; use "
                "'partial' or 'real' if it does"
            )
        return self
