"""Helpers to register tools onto a live ``ToolRegistry``.

Does not implement handlers — CapabilityAuthor still writes code/tests.
Registration makes a descriptor selectable by Conductor in-process.
"""

from __future__ import annotations

from labpilot.research_engine.tools.descriptors import ToolDescriptor
from labpilot.research_engine.tools.registry import ToolRegistry


def register_tool(registry: ToolRegistry, descriptor: ToolDescriptor) -> str:
    """Register (or replace) ``descriptor``; return its name."""
    if not descriptor.name or not descriptor.name.strip():
        raise ValueError("ToolDescriptor.name is required")
    registry.register(descriptor)
    return descriptor.name
