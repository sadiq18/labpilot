"""CapabilityExecutor protocol — tools the Research Engineer dispatches to."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType


@runtime_checkable
class CapabilityExecutor(Protocol):
    """Stable skill surface; does not own the global queue."""

    name: str

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        ...

    def prepare(self, context: TaskContext) -> None:
        ...

    def execute(self, context: TaskContext) -> TaskEvidence:
        ...

    def verify(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        ...

    def rollback(self, context: TaskContext) -> None:
        ...

    def collect_evidence(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        ...


class BaseCapability:
    """Convenience base with no-op prepare/rollback and pass-through verify."""

    name: str = "base"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset()

    def prepare(self, context: TaskContext) -> None:
        return None

    def verify(self, context: TaskContext, evidence: TaskEvidence) -> TaskEvidence:
        return evidence

    def rollback(self, context: TaskContext) -> None:
        return None

    def collect_evidence(
        self, context: TaskContext, evidence: TaskEvidence
    ) -> TaskEvidence:
        return evidence

    def execute(self, context: TaskContext) -> TaskEvidence:  # pragma: no cover - abstract
        raise NotImplementedError
