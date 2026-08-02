"""Research OS tool runtime — named capabilities over artifact adapters.

Callers (CLI / orchestrators) invoke tools by name. Handlers wrap existing
libraries in-process and return artifact refs; they must not chain into the
next research step.

Import direction::

    callers → tools → {artifacts, workspace_facade} → engine packages
    engine packages must NOT import ``labpilot.research_engine.tools``
"""

from __future__ import annotations

from labpilot.research_engine.tools.catalog import build_default_tool_registry
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registration import register_tool
from labpilot.research_engine.tools.registry import ToolRegistry

__all__ = [
    "ToolDescriptor",
    "ToolRegistry",
    "ToolResult",
    "build_default_tool_registry",
    "register_tool",
]
