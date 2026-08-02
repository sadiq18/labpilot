"""Conductor run loop — campaign-aware observe → action → approve → dispatch."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from labpilot.research_engine.conductor.actions import (
    ResearchAction,
    map_research_action,
    offline_next_research_action,
)
from labpilot.research_engine.conductor.approvals import (
    ApprovalPrompt,
    OfflineFallbackPrompt,
    maybe_approve,
)
from labpilot.research_engine.conductor.budgets import evaluate_stops
from labpilot.research_engine.conductor.checkpoint import (
    load_budget_pair,
    persist_budgets,
    save_checkpoint,
)
from labpilot.research_engine.conductor.gap_ledger import build_suggestion_context
from labpilot.research_engine.conductor.metrics import ensure_metrics, record_suggestion
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
    autonomy: int = 0,
    campaign_mode: bool = True,
    prefer_offline: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
) -> list[DecisionRecord]:
    """Run until stop, budget, max_steps, or operator pause status.

    When online policy fails, asks the operator (allow / deny / retry) before
    using the deterministic offline order — unless ``prefer_offline`` or
    ``auto_approve`` (``--yes``) is set.
    """
    scheduler = Scheduler(store, registry, workspace)
    decisions: list[DecisionRecord] = []
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")

    ensure_metrics(store, session_id)
    budget_cfg, budget_state = load_budget_pair(session)
    budget_state.ensure_wall_start()
    persist_budgets(store, session_id, budget_cfg, budget_state)

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    policy_kw: dict[str, Any] = {
        "prefer_offline": prefer_offline,
        "auto_offline_fallback": auto_approve,
        "offline_fallback_prompt": offline_fallback_prompt,
    }

    for step in range(max_steps):
        # Refresh each iteration so mid-session registration is visible.
        allowlist = set(registry.names())
        session = store.get_session(session_id)
        assert session is not None
        if session.status == "paused":
            _progress("Session paused by operator")
            break

        budget_cfg, budget_state = load_budget_pair(session)
        stop = evaluate_stops(budget_cfg, budget_state)
        if stop != "none":
            store.update_session_status(session_id, "completed")
            if stop != "metric_target":
                store.increment_metric(session_id, "unmet_goal")
            _progress(f"Stop condition: {stop}")
            decisions.append(
                DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=f"stop:{stop}",
                    stop=True,
                    observe={"stop_reason": stop},
                )
            )
            store.append_decision(decisions[-1])
            break

        if campaign_mode:
            completed = [
                t.tool_name
                for t in store.list_tasks(session_id)
                if t.status == "completed"
            ]
            if prefer_offline:
                research = offline_next_research_action(completed, allowlist)
            else:
                next_tool, _obs = decide_next(
                    store,
                    workspace,
                    session_id,
                    registry,
                    llm_client=llm_client,
                    **policy_kw,
                )
                if next_tool.stop or not next_tool.tool:
                    research = ResearchAction(
                        intent="stop",
                        rationale=next_tool.rationale or "stop",
                        stop=True,
                    )
                else:
                    research = ResearchAction(
                        intent=f"Run tool {next_tool.tool}",
                        rationale=next_tool.rationale,
                        suggested_tools=[next_tool.tool],
                    )
            # Re-read after policy/offline so same-step registration is visible.
            allowlist = set(registry.names())
            plan = map_research_action(research, allowlist)
            if research.stop:
                record = DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=research.rationale,
                    stop=True,
                )
                store.append_decision(record)
                decisions.append(record)
                store.update_session_status(session_id, "completed")
                _progress(f"Conductor stop: {research.rationale}")
                break
            if plan.unmapped:
                ctx = build_suggestion_context(
                    intent=research.intent,
                    suggested_tools=research.suggested_tools,
                    missing_tools=plan.missing_tools,
                    competition=store.competition,
                    session_id=session_id,
                    goal=session.goal,
                )
                record_suggestion(
                    store,
                    session_id,
                    plan.suggestion or research.intent,
                    context=ctx,
                )
                record = DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=plan.suggestion or "no_capability",
                    stop=False,
                    observe={
                        "unmapped": True,
                        "intent": research.intent,
                        "missing_tools": list(plan.missing_tools),
                    },
                )
                store.append_decision(record)
                decisions.append(record)
                _progress(f"No capability: {plan.suggestion}")
                continue

            prev_id: str | None = None
            for tool_step in plan.steps:
                decision_id = store.new_decision_id()
                deps = [prev_id] if prev_id else []
                task = store.enqueue(
                    session_id,
                    tool_step.tool,
                    args=tool_step.args,
                    decision_id=decision_id,
                    dependencies=deps,
                )
                record = DecisionRecord(
                    id=decision_id,
                    session_id=session_id,
                    tool_name=tool_step.tool,
                    rationale=research.rationale or research.intent,
                    args=tool_step.args,
                    task_id=task.id,
                    observe={"step": step, "intent": research.intent},
                )
                approval = maybe_approve(
                    store,
                    session_id=session_id,
                    tool_name=tool_step.tool,
                    decision_id=decision_id,
                    task_id=task.id,
                    auto=auto_approve,
                    prompt=approval_prompt,
                    autonomy=autonomy,
                )
                if approval is not None:
                    record.approval = approval
                    if approval.decision == "reject":
                        store.update_task_status(
                            task.id, "cancelled", error="operator rejected"
                        )
                        store.increment_metric(session_id, "tasks_blocked")
                        store.append_decision(record)
                        decisions.append(record)
                        _progress(f"Rejected {tool_step.tool}")
                        break
                _progress(f"Dispatch {tool_step.tool} ({task.id})")
                try:
                    result = scheduler.dispatch(task)
                    record.artifact_refs = [r.model_dump() for r in result.refs]
                    if tool_step.tool in {"submit", "submit_learn"}:
                        store.increment_metric(session_id, "submissions")
                        budget_cfg, budget_state = load_budget_pair(
                            store.get_session(session_id)  # type: ignore[arg-type]
                        )
                        budget_state.submissions += 1
                        persist_budgets(store, session_id, budget_cfg, budget_state)
                except Exception as exc:
                    store.increment_metric(session_id, "tasks_failed")
                    record.rationale = f"{record.rationale} | dispatch error: {exc}"
                    store.append_decision(record)
                    decisions.append(record)
                    _progress(f"Task failed: {exc}")
                    break
                store.append_decision(record)
                decisions.append(record)
                prev_id = task.id
            save_checkpoint(store, session_id)
            continue

        # Legacy M2 single-tool path
        action, observe = decide_next(
            store,
            workspace,
            session_id,
            registry,
            llm_client=llm_client,
            **policy_kw,
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
            autonomy=autonomy,
        )
        if approval is not None:
            record.approval = approval
            if approval.decision == "reject":
                store.update_task_status(task.id, "cancelled", error="operator rejected")
                store.append_decision(record)
                decisions.append(record)
                continue
        try:
            result = scheduler.dispatch(task)
            record.artifact_refs = [r.model_dump() for r in result.refs]
        except Exception as exc:
            record.rationale = f"{record.rationale} | dispatch error: {exc}"
            store.append_decision(record)
            decisions.append(record)
            continue
        store.append_decision(record)
        decisions.append(record)
        save_checkpoint(store, session_id)
    else:
        store.update_session_status(session_id, "paused")
        _progress(f"Reached max_steps={max_steps}")
        save_checkpoint(store, session_id, extra={"stop_reason": "max_steps"})

    return decisions
