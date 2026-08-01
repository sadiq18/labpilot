"""Dispatch ready OS tasks to the ToolRegistry."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.conductor.models import OsTask
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace


class Scheduler:
    """Pick the next ready task and invoke its tool. Does not chain tools."""

    def __init__(
        self,
        store: ConductorStore,
        registry: ToolRegistry,
        workspace: Workspace,
    ) -> None:
        self.store = store
        self.registry = registry
        self.workspace = workspace

    def next_ready(self, session_id: str) -> OsTask | None:
        ready = self.store.ready_tasks(session_id)
        return ready[0] if ready else None

    def dispatch(self, task: OsTask) -> ToolResult:
        """Mark running, invoke tool, mark completed/failed."""
        self.store.update_task_status(task.id, "running")
        try:
            result = self.registry.invoke(task.tool_name, self.workspace, **task.args)
        except Exception as exc:
            if task.retry_count < task.max_retries:
                self.store.update_task_status(task.id, "retry", error=str(exc))
            else:
                self.store.update_task_status(task.id, "failed", error=str(exc))
            raise
        refs = [r.model_dump() for r in result.refs]
        self.store.update_task_status(task.id, "completed", artifact_refs=refs)
        return result

    def dispatch_next(self, session_id: str) -> tuple[OsTask, ToolResult] | None:
        task = self.next_ready(session_id)
        if task is None:
            return None
        result = self.dispatch(task)
        return task, result
