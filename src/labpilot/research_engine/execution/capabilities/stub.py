"""No-op stub capability — marks any task done with empty evidence (wiring tests)."""

from __future__ import annotations

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


class StubCapability(BaseCapability):
    """Handles every TaskType; used until real capabilities register.

    `verifies = False` because always passing is what a stub is *for*, and that
    is precisely what made it dangerous: on an evidence card its `passed=True`
    was indistinguishable from a capability that had checked something. Four
    campaigns on 2026-08-08 ran with codegen silently falling back to a template
    and every card reading clean. M20 does not ask a stub to fail; it asks it to
    stop claiming a verification it never performed.
    """

    name = "stub"
    verifies = False

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
            checks=["stub_no_verification"],
            summary=f"stub ran {context.task.type} — nothing was verified",
        )
