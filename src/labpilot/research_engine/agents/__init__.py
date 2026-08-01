"""Research OS specialist agent runtime.

Conductor-scheduled skill bundles — not peer autonomous managers.

Import direction::

    Conductor / CLI → agents → {context, workspace, tools, execution(V1)}
    Agents must NOT import peer agents or ``conductor`` for control flow.
"""

from __future__ import annotations

from labpilot.research_engine.agents.catalog import build_default_specialist_registry
from labpilot.research_engine.agents.coding import V1CodeEngineeringCodingTool
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.facade import execute_agent_sync
from labpilot.research_engine.agents.implementation import ImplementationSpecialist
from labpilot.research_engine.agents.models import AgentTask, SpecialistDescriptor, as_agent_task
from labpilot.research_engine.agents.ports import Agent, CodingTool
from labpilot.research_engine.agents.registry import SpecialistRegistry

__all__ = [
    "Agent",
    "AgentTask",
    "CodingTool",
    "ExperimentSpecialist",
    "ImplementationSpecialist",
    "SpecialistDescriptor",
    "SpecialistRegistry",
    "V1CodeEngineeringCodingTool",
    "as_agent_task",
    "build_default_specialist_registry",
    "execute_agent_sync",
]
