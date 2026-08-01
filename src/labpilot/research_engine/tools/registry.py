"""Name-keyed registry for Research OS tools."""

from __future__ import annotations

from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.workspace_facade import Workspace


class ToolRegistry:
    """Register and invoke tools by stable name.

    Handlers return artifacts (or refs). They must not call another tool to
    sequence the next research step — callers own orchestration.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        """Add or replace a tool under ``descriptor.name``."""
        self._tools[descriptor.name] = descriptor

    def get(self, name: str) -> ToolDescriptor | None:
        """Return a descriptor by name, or ``None``."""
        return self._tools.get(name)

    def require(self, name: str) -> ToolDescriptor:
        """Return a descriptor or raise :class:`KeyError`."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"no tool registered: {name}")
        return tool

    def list_tools(self) -> list[ToolDescriptor]:
        """Return registered tools sorted by name."""
        return [self._tools[k] for k in sorted(self._tools)]

    def names(self) -> list[str]:
        """Return sorted tool names."""
        return sorted(self._tools)

    def invoke(self, name: str, workspace: Workspace, **params: object) -> ToolResult:
        """Run a tool against ``workspace`` with keyword parameters."""
        tool = self.require(name)
        missing = [
            field
            for field in tool.required_workspace_fields
            if not getattr(workspace, field, None)
        ]
        if missing:
            raise ValueError(
                f"tool {name!r} requires workspace fields: {', '.join(missing)}"
            )
        return tool.handler(workspace, **params)
