"""No-op stub capability — marks any task done with empty evidence (wiring tests)."""

from __future__ import annotations

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class StubCapability(BaseCapability):
    """Handles every TaskType; used until real capabilities register."""

    name = "stub"

    def __init__(self, task_types: frozenset[TaskType] | None = None) -> None:
        self._types = task_types or frozenset(TaskType)

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return self._types

    def execute(self, context: TaskContext) -> TaskEvidence:
        return TaskEvidence(
            task_id=context.task.id,
            execution_id=context.execution.id,
            capability=self.name,
            passed=True,
            summary=f"stub completed {context.task.type}",
        )
