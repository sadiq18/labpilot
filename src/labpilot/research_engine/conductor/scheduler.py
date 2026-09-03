"""Dispatch ready OS tasks to the ToolRegistry."""

from __future__ import annotations

import inspect
from typing import Any

from labpilot.research_engine.conductor.models import OsTask
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace


def with_llm_client(
    registry: ToolRegistry,
    tool_name: str,
    args: dict[str, Any],
    llm_client: Any | None,
) -> dict[str, Any]:
    """Add `llm_client` to `args` only when the handler declares the parameter.

    Signature-driven rather than a hardcoded tool list, so a new execution tool
    is wired by accepting `llm_client` — the failure mode being avoided is a
    tool silently running without one.

    Module-level because the Conductor is no longer the only caller: M16's
    evidence producer invokes a tool outside the task queue, and a second copy
    of this rule is a second place for it to fall out of date.
    """
    if llm_client is None or "llm_client" in args:
        return args
    tool = registry.get(tool_name)
    handler = getattr(tool, "handler", None)
    if handler is None:
        return args
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return args
    if "llm_client" not in params:
        return args
    return {**args, "llm_client": llm_client}


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
        """This scheduler's client, through the shared rule above."""
        return with_llm_client(self.registry, tool_name, args, self.llm_client)

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
            # `failed`, not `retry`. Marking `retry` was a promise nobody kept:
            # `update_task_status` turns `retry` into `pending`, and nothing
            # re-dispatches a pending task — `dispatch_next` and `next_ready`
            # have no callers, and the campaign loop dispatches the task it just
            # enqueued. So a task that blew up was parked as `pending` for good
            # while every other layer recorded the failure correctly: the
            # `tasks_failed` metric, the breaker's execution counters, and the
            # decision record all agreed it failed, and only the task row said
            # it was still waiting to run.
            #
            # That row is not private bookkeeping. The observe bundle sends
            # `task_summary` to the policy every step, so a campaign's failures
            # accumulated there reading as queued work.
            #
            # Retrying is not the missing piece. The campaign's retry *is* the
            # next policy step, which re-decides with the error in context —
            # better than a blind re-run, and a blind re-run of `run_experiment`
            # means training the model again to watch it fail the same way.
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
