"""Research Engineer — deterministic plan orchestrator.

Owns queue, dispatch, evidence, verification, recovery, and resume.
Never calls an LLM to choose the next task.
"""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.evidence import read_evidence, write_evidence
from labpilot.research_engine.execution.recovery import RecoveryAction, decide_recovery
from labpilot.research_engine.execution.registry import CapabilityRegistry
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.execution.store import (
    ExecutionStore,
    competition_workspace_path,
)
from labpilot.research_engine.execution.verification import verify_evidence
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskStatus
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import topological_levels, validate_plan

logger = logging.getLogger(__name__)


class EngineerError(RuntimeError):
    """Fatal orchestration failure."""


class ResearchEngineer:
    """Walk an approved ResearchPlan via registered capabilities."""

    def __init__(
        self,
        *,
        knowledge_dir: Path,
        competition: str,
        registry: CapabilityRegistry,
        plan_store: PlanStore | None = None,
        execution_store: ExecutionStore | None = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.competition = competition
        self.registry = registry
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self._plan_store = plan_store or PlanStore(knowledge_dir, competition)
        self._owns_plan_store = plan_store is None
        self._exec_store = execution_store or ExecutionStore(knowledge_dir, competition)
        self._owns_exec_store = execution_store is None

    def close(self) -> None:
        if self._owns_plan_store:
            self._plan_store.close()
        if self._owns_exec_store:
            self._exec_store.close()

    def run_plan(self, plan_id: str) -> ResearchExecution:
        """Create an execution and run all pending tasks to completion."""
        plan = self._load_runnable_plan(plan_id)
        execution = self._exec_store.create_execution(plan_id)
        self._plan_store.update_plan_status(plan_id, PlanStatus.IN_PROGRESS)
        self._exec_store.update_status(execution.id, "running")
        return self._run_execution(execution.id)

    def resume(self, execution_id: str) -> ResearchExecution:
        """Continue from the first non-terminal task."""
        execution = self._exec_store.get_execution(execution_id)
        if execution is None:
            raise EngineerError(f"unknown execution_id: {execution_id}")
        if execution.status in {"succeeded", "cancelled"}:
            return execution
        self._exec_store.update_status(execution_id, "running")
        plan = self._plan_store.get_plan(execution.plan_id)
        if plan is not None and plan.status != PlanStatus.IN_PROGRESS:
            self._plan_store.update_plan_status(execution.plan_id, PlanStatus.IN_PROGRESS)
        return self._run_execution(execution_id)

    def _run_execution(self, execution_id: str) -> ResearchExecution:
        execution = self._exec_store.get_execution(execution_id)
        assert execution is not None
        plan = self._plan_store.get_plan(execution.plan_id)
        if plan is None:
            raise EngineerError(f"plan missing for execution {execution_id}")

        try:
            for level in topological_levels(plan):
                for task_id in level:
                    # Reload plan so status updates are visible.
                    plan = self._plan_store.get_plan(execution.plan_id)
                    assert plan is not None
                    task = next(t for t in plan.tasks if t.id == task_id)
                    if task.status in {
                        TaskStatus.DONE,
                        TaskStatus.SKIPPED,
                        TaskStatus.FAILED,
                    }:
                        if task.status == TaskStatus.FAILED:
                            raise EngineerError(
                                f"task {task.id} already failed; cannot continue"
                            )
                        continue
                    self._run_task(plan, execution_id, task_id)
        except EngineerError as exc:
            self._exec_store.update_status(execution_id, "failed", error=str(exc))
            self._plan_store.update_plan_status(execution.plan_id, PlanStatus.ABANDONED)
            result = self._exec_store.get_execution(execution_id)
            assert result is not None
            return result

        self._exec_store.update_status(execution_id, "succeeded")
        self._plan_store.update_plan_status(execution.plan_id, PlanStatus.DONE)
        result = self._exec_store.get_execution(execution_id)
        assert result is not None
        return result

    def _run_task(
        self,
        plan: ResearchPlan,
        execution_id: str,
        task_id: str,
    ) -> None:
        plan = self._plan_store.get_plan(plan.id)
        assert plan is not None
        task = next(t for t in plan.tasks if t.id == task_id)
        execution = self._exec_store.get_execution(execution_id)
        assert execution is not None

        capability = self.registry.require(task.type)
        attempt = int(task.metadata.get("attempt", 0))
        workspace = Path(
            execution.workspace_path
            or competition_workspace_path(self.knowledge_dir, self.competition)
        )

        while True:
            self._plan_store.update_task_status(
                task.id,
                TaskStatus.RUNNING,
                metadata_patch={"capability": capability.name, "attempt": attempt},
            )
            prior = read_evidence(self.paths, execution_id, task.id)
            context = TaskContext(
                plan=plan,
                task=task,
                execution=execution,
                paths=self.paths,
                workspace_root=workspace,
                competition=self.competition,
                prior_evidence=prior,
                runtime_target=execution.runtime_target,
                attempt=attempt,
            )
            capability.prepare(context)
            evidence = capability.execute(context)
            evidence = capability.verify(context, evidence)
            evidence = capability.collect_evidence(context, evidence)
            write_evidence(self.paths, evidence)

            if verify_evidence(evidence):
                self._plan_store.update_task_status(
                    task.id,
                    TaskStatus.DONE,
                    metadata_patch={
                        "capability": capability.name,
                        "attempt": attempt,
                        "evidence": str(
                            self.paths.executions_dir
                            / execution_id
                            / "evidence"
                            / f"{task.id}.json"
                        ),
                    },
                )
                return

            decision = decide_recovery(task, evidence, attempt=attempt)
            if decision.action == RecoveryAction.RETRY:
                attempt += 1
                logger.info(
                    "Retrying task %s (%s)", task.id, decision.reason
                )
                continue
            if decision.action == RecoveryAction.SKIP:
                self._plan_store.update_task_status(
                    task.id,
                    TaskStatus.SKIPPED,
                    error=decision.reason,
                    metadata_patch={"attempt": attempt},
                )
                return

            self._plan_store.update_task_status(
                task.id,
                TaskStatus.FAILED,
                error=decision.reason,
                metadata_patch={"attempt": attempt},
            )
            raise EngineerError(
                f"task {task.id} failed: {decision.reason}"
            )

    def _load_runnable_plan(self, plan_id: str) -> ResearchPlan:
        plan = self._plan_store.get_plan(plan_id)
        if plan is None:
            raise EngineerError(f"unknown plan_id: {plan_id}")
        if plan.status not in {PlanStatus.READY, PlanStatus.IN_PROGRESS}:
            raise EngineerError(
                f"plan {plan_id} status={plan.status}; need ready or in_progress"
            )
        validate_plan(plan)
        return plan


def default_stub_registry() -> CapabilityRegistry:
    """Registry with a stub covering every TaskType (Plan 2 wiring)."""
    from labpilot.research_engine.execution.capabilities.stub import StubCapability

    registry = CapabilityRegistry()
    registry.register(StubCapability())
    return registry


def default_capability_registry(*, install_packages: bool = True) -> CapabilityRegistry:
    """Production-leaning registry: workspace + dependency, stub for the rest."""
    from labpilot.research_engine.execution.capabilities.dependency import (
        DependencyCapability,
    )
    from labpilot.research_engine.execution.capabilities.stub import StubCapability
    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    registry = CapabilityRegistry()
    # Stub first; real capabilities override overlapping types.
    covered = frozenset(
        {
            TaskType.PREPARE_WORKSPACE,
            TaskType.INSTALL_PACKAGE,
        }
    )
    registry.register(StubCapability(frozenset(TaskType) - covered))
    registry.register(WorkspaceCapability())
    registry.register(DependencyCapability(install=install_packages))
    return registry
