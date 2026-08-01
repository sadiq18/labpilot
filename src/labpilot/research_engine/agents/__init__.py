"""Research OS specialist agent runtime.

Conductor-scheduled skill bundles — not peer autonomous managers.

Import direction::

    Conductor / CLI → agents → {context, workspace, tools, execution(V1)}
    Agents must NOT import peer agents or ``conductor`` for control flow.
"""

from __future__ import annotations

from labpilot.research_engine.agents.catalog import build_default_specialist_registry
from labpilot.research_engine.agents.coding import V1CodeEngineeringCodingTool
from labpilot.research_engine.agents.events import (
    EVIDENCE_UPDATED,
    EXPERIMENT_COMPLETED,
    IMPLEMENTATION_FINISHED,
    MODEL_FAILED,
    EventBus,
    EventEmitter,
    default_event_bus,
    noop_emit,
)
from labpilot.research_engine.agents.experiment import ExperimentSpecialist
from labpilot.research_engine.agents.facade import execute_agent_sync
from labpilot.research_engine.agents.implementation import ImplementationSpecialist
from labpilot.research_engine.agents.models import AgentTask, SpecialistDescriptor, as_agent_task
from labpilot.research_engine.agents.parallel import (
    ParallelBudget,
    ParallelResult,
    ParallelWorkItem,
    parallel_summary,
    run_parallel_async,
    run_parallel_sync,
)
from labpilot.research_engine.agents.ports import Agent, CodingTool
from labpilot.research_engine.agents.registry import SpecialistRegistry
from labpilot.research_engine.agents.subscribers import install_evidence_refresh_subscriber

__all__ = [
    "EVIDENCE_UPDATED",
    "EXPERIMENT_COMPLETED",
    "IMPLEMENTATION_FINISHED",
    "MODEL_FAILED",
    "Agent",
    "AgentTask",
    "CodingTool",
    "EventBus",
    "EventEmitter",
    "ExperimentSpecialist",
    "ImplementationSpecialist",
    "ParallelBudget",
    "ParallelResult",
    "ParallelWorkItem",
    "SpecialistDescriptor",
    "SpecialistRegistry",
    "V1CodeEngineeringCodingTool",
    "as_agent_task",
    "build_default_specialist_registry",
    "default_event_bus",
    "execute_agent_sync",
    "install_evidence_refresh_subscriber",
    "noop_emit",
    "parallel_summary",
    "run_parallel_async",
    "run_parallel_sync",
]
