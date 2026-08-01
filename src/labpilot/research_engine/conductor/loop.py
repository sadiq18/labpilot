"""Conductor run loop — observe → think → enqueue → approve → dispatch → log."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from labpilot.research_engine.conductor.approvals import ApprovalPrompt, maybe_approve
from labpilot.research_engine.conductor.models import DecisionRecord
from labpilot.research_engine.conductor.policy import decide_next
from labpilot.research_engine.conductor.scheduler import Scheduler
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


def run_until_stop(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
    max_steps: int = 8,
    auto_approve: bool = False,
    approval_prompt: ApprovalPrompt | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[DecisionRecord]:
    """Run the Conductor loop until stop, max_steps, or session failure."""
    scheduler = Scheduler(store, registry, workspace)
    decisions: list[DecisionRecord] = []
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    for step in range(max_steps):
        action, observe = decide_next(
            store, workspace, session_id, registry, llm_client=llm_client
        )
        decision_id = store.new_decision_id()
        record = DecisionRecord(
            id=decision_id,
            session_id=session_id,
            tool_name=action.tool,
            rationale=action.rationale,
            stop=action.stop,
            args=action.args,
            observe={
                "step": step,
                "completed_tools": observe.get("completed_tools"),
                "operator_feedback": observe.get("operator_feedback"),
            },
        )

        if action.stop or not action.tool:
            store.append_decision(record)
            decisions.append(record)
            store.update_session_status(session_id, "completed")
            _progress(f"Conductor stop: {action.rationale}")
            break

        task = store.enqueue(
            session_id,
            action.tool,
            args=action.args,
            decision_id=decision_id,
        )
        record.task_id = task.id

        approval = maybe_approve(
            store,
            session_id=session_id,
            tool_name=action.tool,
            decision_id=decision_id,
            task_id=task.id,
            auto=auto_approve,
            prompt=approval_prompt,
        )
        if approval is not None:
            record.approval = approval
            if approval.decision == "reject":
                store.update_task_status(task.id, "cancelled", error="operator rejected")
                store.append_decision(record)
                decisions.append(record)
                _progress(
                    f"Rejected {action.tool}"
                    + (f": {approval.comment}" if approval.comment else "")
                )
                continue

        _progress(f"Dispatch {action.tool} ({task.id})")
        try:
            result = scheduler.dispatch(task)
            record.artifact_refs = [r.model_dump() for r in result.refs]
        except Exception as exc:
            record.rationale = f"{record.rationale} | dispatch error: {exc}"
            store.append_decision(record)
            decisions.append(record)
            _progress(f"Task failed: {exc}")
            # Continue loop so policy can recover.
            continue

        store.append_decision(record)
        decisions.append(record)
    else:
        store.update_session_status(session_id, "paused")
        _progress(f"Reached max_steps={max_steps}")

    return decisions
