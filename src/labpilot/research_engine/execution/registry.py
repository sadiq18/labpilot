"""CapabilityRegistry — map TaskType → CapabilityExecutor."""

from __future__ import annotations

from labpilot.research_engine.execution.capabilities.base import CapabilityExecutor
from labpilot.research_engine.planner.schemas.task_types import TaskType


class CapabilityRegistry:
    def __init__(self) -> None:
        self._by_type: dict[TaskType, CapabilityExecutor] = {}
        self._capabilities: list[CapabilityExecutor] = []

    def register(self, capability: CapabilityExecutor) -> None:
        self._capabilities.append(capability)
        for task_type in capability.supported_task_types:
            self._by_type[task_type] = capability

    def get(self, task_type: TaskType) -> CapabilityExecutor | None:
        return self._by_type.get(task_type)

    def require(self, task_type: TaskType) -> CapabilityExecutor:
        cap = self.get(task_type)
        if cap is None:
            raise KeyError(f"no capability registered for task type: {task_type}")
        return cap

    @property
    def capabilities(self) -> list[CapabilityExecutor]:
        return list(self._capabilities)
