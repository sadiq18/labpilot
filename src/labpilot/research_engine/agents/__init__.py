"""Research OS specialist agent runtime.

Conductor-scheduled skill bundles — not peer autonomous managers.

Import direction::

    Conductor / CLI → agents → {context, workspace, tools, execution(V1)}
    Agents must NOT import peer agents or ``conductor`` for control flow.
"""

from __future__ import annotations

from labpilot.research_engine.agents.coding import V1CodeEngineeringCodingTool
from labpilot.research_engine.agents.facade import execute_agent_sync
from labpilot.research_engine.agents.models import AgentTask, SpecialistDescriptor
from labpilot.research_engine.agents.ports import Agent, CodingTool
from labpilot.research_engine.agents.registry import SpecialistRegistry

__all__ = [
    "Agent",
    "AgentTask",
    "CodingTool",
    "SpecialistDescriptor",
    "SpecialistRegistry",
    "V1CodeEngineeringCodingTool",
    "execute_agent_sync",
]
