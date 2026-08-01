"""CodingTool backends — V1 Code Engineering is the default."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import anyio

from labpilot.research_engine.agents.models import AgentTask
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.workspace_facade import Workspace

_CODE_SCHEMA = "labpilot.artifact.code/v1"

_CAPABILITY_TO_TASK_TYPE: dict[str, TaskType] = {
    "implement": TaskType.WRITE_CODE,
    "write_code": TaskType.WRITE_CODE,
    "write": TaskType.WRITE_CODE,
    "read_code": TaskType.READ_CODE,
    "read": TaskType.READ_CODE,
    "modify_config": TaskType.MODIFY_CONFIG,
}


def _as_agent_task(task: object) -> AgentTask:
    if isinstance(task, AgentTask):
        return task
    if isinstance(task, dict):
        return AgentTask.model_validate(task)
    task_id = str(getattr(task, "id", None) or getattr(task, "task_id", None) or "T-coding")
    capability = str(
        getattr(task, "capability", None)
        or getattr(task, "tool_name", None)
        or "implement"
    )
    description = str(getattr(task, "description", None) or getattr(task, "goal", None) or "")
    metadata = getattr(task, "metadata", None) or getattr(task, "args", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return AgentTask(
        id=task_id,
        capability=capability,
        description=description,
        metadata=dict(metadata),
    )


def _task_type_for(agent_task: AgentTask) -> TaskType:
    raw = str(agent_task.metadata.get("task_type") or agent_task.capability).strip().lower()
    if raw in _CAPABILITY_TO_TASK_TYPE:
        return _CAPABILITY_TO_TASK_TYPE[raw]
    try:
        return TaskType(raw)
    except ValueError:
        return TaskType.WRITE_CODE


def build_v1_task_context(
    workspace: Workspace,
    agent_task: AgentTask,
    context: ContextBundle,
) -> TaskContext:
    """Assemble a minimal V1 :class:`TaskContext` from Workspace + AgentTask."""
    competition = workspace.competition or context.request.competition
    paths = ResearchPaths(workspace.knowledge_dir, competition).ensure()
    now = datetime.now(UTC)
    task_type = _task_type_for(agent_task)
    plan = ResearchPlan(
        id=f"P-agent-{agent_task.id}",
        competition=competition,
        hypothesis_id="",
        goal=agent_task.description or context.request.goal or "implement",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id=agent_task.id,
                plan_id=f"P-agent-{agent_task.id}",
                type=task_type,
                description=agent_task.description,
                metadata=dict(agent_task.metadata),
            )
        ],
        created_at=now,
        updated_at=now,
    )
    execution = ResearchExecution(
        id=f"E-agent-{agent_task.id}",
        plan_id=plan.id,
        competition=competition,
        status="running",
        workspace_path=str(workspace.root),
        created_at=now,
        updated_at=now,
    )
    constraints: dict[str, Any] = {}
    if context.summary():
        constraints["context_summary"] = context.summary(max_chars=2000)
    constraints.update(dict(agent_task.metadata.get("constraints") or {}))
    return TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=execution,
        paths=paths,
        workspace_root=workspace.root,
        competition=competition,
        constraints=constraints,
    )


def _refs_from_evidence(
    paths: list[str],
    *,
    competition: str,
    task_id: str,
) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for i, path in enumerate(paths):
        refs.append(
            ArtifactRef(
                kind="code",
                id=f"code:{task_id}:{i}",
                schema_id=_CODE_SCHEMA,
                path=path,
                competition=competition,
            )
        )
    return refs


class V1CodeEngineeringCodingTool:
    """CodingTool backend: existing :class:`CodeEngineeringCapability`."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._capability = CodeEngineeringCapability(llm_client=llm_client)

    async def implement(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        agent_task = _as_agent_task(task)
        task_ctx = build_v1_task_context(workspace, agent_task, context)
        evidence = await anyio.to_thread.run_sync(self._capability.execute, task_ctx)
        if not evidence.passed:
            raise RuntimeError(evidence.error or evidence.summary or "coding failed")
        return _refs_from_evidence(
            list(evidence.paths),
            competition=workspace.competition,
            task_id=agent_task.id,
        )
