"""Dispatch ready OS tasks to the ToolRegistry."""

from __future__ import annotations

import inspect
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
        llm_client: Any | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.workspace = workspace
        # Task args are persisted JSON and cannot carry a live client, so the
        # only way an execution tool receives one is from here. Without this the
        # LLM reached the Conductor's *policy* (`decide_next`) but never its
        # *execution* path: `run_plan` built a CodeEngineeringCapability with
        # `llm_client=None`, and M14 phase 2a then refused the whole run.
        #
        # Before 2a that refusal was a silent degrade to the rule engine, which
        # returns `files=[]` — straight to the template fallback and the same
        # baseline every time. It is one of the mechanisms behind MSE 194.80
        # repeating twelve times.
        self.llm_client = llm_client

    def _with_llm_client(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Inject the client only into handlers that declare the parameter.

        Signature-driven rather than a hardcoded tool list, so a new execution
        tool is wired by accepting `llm_client` — the failure mode being avoided
        is a tool silently running without one.
        """
        if self.llm_client is None or "llm_client" in args:
            return args
        tool = self.registry.get(tool_name)
        handler = getattr(tool, "handler", None)
        if handler is None:
            return args
        try:
            params = inspect.signature(handler).parameters
        except (TypeError, ValueError):
            return args
        if "llm_client" not in params:
            return args
        return {**args, "llm_client": self.llm_client}

    def next_ready(self, session_id: str) -> OsTask | None:
        ready = self.store.ready_tasks(session_id)
        return ready[0] if ready else None

    def dispatch(self, task: OsTask) -> ToolResult:
        """Mark running, invoke tool, mark completed/failed."""
        self.store.update_task_status(task.id, "running")
        try:
            args = self._with_llm_client(task.tool_name, dict(task.args))
            result = self.registry.invoke(task.tool_name, self.workspace, **args)
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
