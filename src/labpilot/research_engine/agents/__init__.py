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
from labpilot.research_engine.agents.git_evolution import (
    find_experiment_record,
    revert_to_commit,
    snapshot_before_experiment,
    write_experiment_git_record,
)
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
from labpilot.research_engine.agents.subscribers import (
    install_default_subscribers,
    install_evidence_refresh_subscriber,
)
from labpilot.research_engine.memory.hooks import install_experience_memory_subscriber
from labpilot.research_engine.git import (
    CODE_PATHS,
    CommitSnapshot,
    GitTool,
    open_git_tool,
)

__all__ = [
    "CODE_PATHS",
    "EVIDENCE_UPDATED",
    "EXPERIMENT_COMPLETED",
    "IMPLEMENTATION_FINISHED",
    "MODEL_FAILED",
    "Agent",
    "AgentTask",
    "CodingTool",
    "CommitSnapshot",
    "EventBus",
    "EventEmitter",
    "ExperimentSpecialist",
    "GitTool",
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
    "find_experiment_record",
    "install_default_subscribers",
    "install_evidence_refresh_subscriber",
    "install_experience_memory_subscriber",
    "noop_emit",
    "open_git_tool",
    "parallel_summary",
    "revert_to_commit",
    "run_parallel_async",
    "run_parallel_sync",
    "snapshot_before_experiment",
    "write_experiment_git_record",
]
