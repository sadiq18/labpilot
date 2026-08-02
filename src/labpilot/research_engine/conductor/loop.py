"""Conductor run loop — campaign-aware observe → action → approve → dispatch."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from labpilot.research_engine.conductor.actions import (
    ResearchAction,
    map_research_action,
    offline_next_research_action,
    resolve_step_args,
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

# How many times an advisory "stop" is overridden while the objective is unmet.
# Bounded so a policy that genuinely has nothing left can still end the run.
_MAX_STOP_OVERRIDES = 2


def _objective_unmet(config: Any, state: Any) -> bool:
    """True when a metric target was set and the best result has not reached it."""
    target = getattr(config, "target_value", None)
    if getattr(config, "target_metric", None) is None or target is None:
        return False
    last = getattr(state, "last_metric", None)
    if last is None:
        return True
    return last < target if getattr(config, "maximize", False) else last > target


def _latest_plan_id(workspace: Workspace) -> str | None:
    """Highest-numbered plan for this competition, or None when none exist."""
    from labpilot.research_engine.artifacts.plan import PlanArtifacts

    artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        plans = artifacts.list()
    except Exception:  # noqa: BLE001 — absent store simply means "no plans yet"
        return None
    finally:
        artifacts.close()
    ids = sorted(p.id for p in plans)
    return ids[-1] if ids else None


def _next_hypothesis_id(workspace: Workspace) -> str | None:
    """Highest-confidence untested hypothesis, or None when there are none.

    This is what lets a campaign iterate: once the baseline plan exists, the
    next plan has to be built against a hypothesis rather than re-requesting an
    idempotent baseline.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    try:
        store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:  # noqa: BLE001 — absent store means "nothing to test yet"
        return None
    if not proposed:
        return None
    ranked = sorted(
        proposed,
        key=lambda h: (getattr(h, "confidence", 0.0) or 0.0, h.id),
        reverse=True,
    )
    return ranked[0].id


def _baseline_plan_exists(workspace: Workspace) -> bool:
    """True when a baseline plan has already been compiled for this competition."""
    from labpilot.research_engine.artifacts.plan import PlanArtifacts

    artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        plans = artifacts.list()
    except Exception:  # noqa: BLE001
        return False
    finally:
        artifacts.close()
    return any((p.metadata or {}).get("plan_kind") == "baseline" for p in plans)


def _latest_execution_id(workspace: Workspace) -> str | None:
    """Most recent execution id, or None when nothing has run yet."""
    from labpilot.research_engine.shared.experiments.graph import build_graph

    try:
        graph = build_graph(
            workspace.effective_runs_dir,
            workspace.competition,
            knowledge_dir=workspace.knowledge_dir,
        )
    except Exception:  # noqa: BLE001
        return None
    nodes = sorted(graph.nodes.values(), key=lambda e: e.created_at)
    return nodes[-1].id if nodes else None


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

    consecutive_stop_overrides = 0
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
                # Observe (Context Engine retrieval) + think (LLM) both run
                # before the first dispatch message, so without this a campaign
                # step is several silent minutes on a local model.
                _progress(f"step {step + 1}/{max_steps}: observing + deciding …")
                started = time.monotonic()
                next_tool, _obs = decide_next(
                    store,
                    workspace,
                    session_id,
                    registry,
                    llm_client=llm_client,
                    **policy_kw,
                )
                _progress(
                    f"step {step + 1}/{max_steps}: chose "
                    f"{next_tool.tool or 'stop'} ({time.monotonic() - started:.1f}s)"
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
            if research.stop and _objective_unmet(budget_cfg, budget_state):
                # Goal persistence. The policy tends to call it done once it has
                # used each tool once ("no immediate next step in the
                # allowlist"), even with the target metric far away. A campaign
                # exists to pursue an objective, so an advisory stop is not
                # honoured while the target is unmet and budget remains —
                # reflection and the next hypothesis are still open moves.
                # Budgets, max_steps and repeated insistence still end the run.
                consecutive_stop_overrides += 1
                if consecutive_stop_overrides <= _MAX_STOP_OVERRIDES:
                    _progress(
                        f"Policy wanted to stop with the objective unmet "
                        f"({consecutive_stop_overrides}/{_MAX_STOP_OVERRIDES}); "
                        "continuing toward the target."
                    )
                    record_suggestion(
                        store,
                        session_id,
                        "Policy stopped early with the objective unmet: "
                        f"{research.rationale}",
                        context={"step": step},
                    )
                    research = ResearchAction(
                        intent="reflect on the last experiment and try the next hypothesis",
                        rationale="objective still unmet; continuing",
                    )
                    plan = map_research_action(research, allowlist)
            else:
                consecutive_stop_overrides = 0

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
                # Resolve @latest against state *now*, after any earlier step in
                # this batch created a plan or execution.
                step_args = resolve_step_args(
                    tool_step.tool,
                    tool_step.args,
                    latest_plan_id=_latest_plan_id(workspace),
                    latest_execution_id=_latest_execution_id(workspace),
                    next_hypothesis_id=_next_hypothesis_id(workspace),
                    baseline_plan_exists=_baseline_plan_exists(workspace),
                )
                task = store.enqueue(
                    session_id,
                    tool_step.tool,
                    args=step_args,
                    decision_id=decision_id,
                    dependencies=deps,
                )
                record = DecisionRecord(
                    id=decision_id,
                    session_id=session_id,
                    tool_name=tool_step.tool,
                    rationale=research.rationale or research.intent,
                    args=step_args,
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
