"""Conductor run loop — campaign-aware observe → action → approve → dispatch."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from labpilot.accessor.common.micro_agents import LLMDegradedError
from labpilot.research_engine.conductor.actions import (
    ResearchAction,
    map_research_action,
    offline_next_research_action,
    resolve_step_args,
)
from labpilot.research_engine.conductor.approvals import (
    SUBMIT_TOOLS,
    ApprovalPrompt,
    OfflineFallbackPrompt,
    maybe_approve,
)
from labpilot.research_engine.conductor.budgets import evaluate_stops, submit_tools_allowed
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
from labpilot.research_engine.planner.schemas.task_types import is_runnable_plan_status
from labpilot.research_engine.telemetry.agent_provenance import recording_provenance
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

# How many times an advisory "stop" is overridden while the objective is unmet.
# Bounded so a policy that genuinely has nothing left can still end the run.
_MAX_STOP_OVERRIDES = 2


#: Tools whose outcome the circuit breaker counts. Only these produce (or fail
#: to produce) an experiment; a failed `analyze_competition` is a bad step, not
#: evidence that the campaign cannot work.
_EXPERIMENT_TOOLS = frozenset({"run_experiment", "run_plan"})


#: Execution statuses that mean the experiment actually produced a result.
#: `pending` and `running` are not successes — an execution left mid-flight has
#: produced nothing, and treating it as a win would reset the breaker on a
#: campaign that is stalling rather than progressing.
_SUCCESS_STATUSES = frozenset({"succeeded"})


def _experiment_outcome(result: object) -> tuple[bool, str]:
    """`(succeeded, error)` for an experiment tool that returned without raising.

    A tool call returning is not an experiment succeeding: `run_plan` reports a
    failed execution in `data["status"]` and raises nothing at all. Reading the
    status is what makes this the question the breaker is supposed to ask.

    An absent status is treated as success, deliberately — this runs on the
    path where the call *worked*, and inventing failures from a missing field
    would stop campaigns for a reporting gap rather than a real one.
    """
    data = getattr(result, "data", None)
    if not isinstance(data, dict) or "status" not in data:
        return True, ""
    status = str(data.get("status") or "").strip().lower()
    if status in _SUCCESS_STATUSES:
        return True, ""
    error = str(data.get("error") or "").strip()
    return False, error or f"execution status={status or 'unknown'}"


def _record_experiment_outcome(
    store,
    session_id: str,
    *,
    succeeded: bool,
    error: str = "",
) -> None:
    """Fold one experiment outcome into the breaker's counters and persist."""
    session = store.get_session(session_id)
    if session is None:
        return
    budget_cfg, budget_state = load_budget_pair(session)
    budget_state.record_execution(succeeded=succeeded, error=error)
    persist_budgets(store, session_id, budget_cfg, budget_state)


def _fail_session_on_degraded_llm(store, session_id, record, decisions, exc) -> None:
    """Record a strict-mode abort before it propagates.

    The session is marked so a later `conduct continue` sees why it ended
    rather than finding a run that simply stopped short.
    """
    record.rationale = f"{record.rationale} | strict LLM abort: {exc}"
    store.append_decision(record)
    decisions.append(record)
    try:
        store.increment_metric(session_id, "tasks_failed")
        store.update_session_status(session_id, "failed")
    except Exception:  # noqa: BLE001 — the abort matters more than the bookkeeping
        logger.warning("could not mark session %s failed", session_id)


def _objective_unmet(config: Any, state: Any) -> bool:
    """True when a metric target was set and the best result has not reached it."""
    target = getattr(config, "target_value", None)
    if getattr(config, "target_metric", None) is None or target is None:
        return False
    last = getattr(state, "last_metric", None)
    if last is None:
        return True
    # `getattr(..., True)` matches `BudgetConfig.maximize`'s own default. This
    # read used `False`, so the two disagreed about the same field whenever the
    # attribute was missing — one more place where direction was assumed rather
    # than resolved.
    return last < target if getattr(config, "maximize", True) else last > target


def _latest_plan_id(workspace: Workspace) -> str | None:
    """Latest plan the Engineer would accept, or None.

    Taking the highest id outright targeted plans that had already finished —
    the Engineer then refused with "status=done; need ready or in_progress" and
    the campaign lost a step. The first fix narrowed the *preference* but kept a
    fallback to the newest plan of any status, so the refusal still happened
    whenever every plan was done. There is no useful answer in that case:
    returning None lets the caller offer `generate_plan` instead of burning a
    step on a run that cannot succeed.
    """
    from labpilot.research_engine.artifacts.plan import PlanArtifacts

    artifacts = PlanArtifacts(workspace.knowledge_dir, workspace.competition)
    try:
        plans = artifacts.list()
    except Exception:  # noqa: BLE001 — absent store simply means "no plans yet"
        return None
    finally:
        artifacts.close()
    if not plans:
        return None
    # A plan testing a retired hypothesis is not work worth dispatching. The
    # campaign selects *plans*, and retiring the idea behind one leaves the plan
    # runnable — measured 2026-08-09, `H-051` was correctly rejected and the
    # very next step selected `P-021`, which carries it.
    from labpilot.research_engine.intelligence.hypothesis.viability import (
        plan_is_selectable,
        retired_hypothesis_ids,
    )

    retired = retired_hypothesis_ids(workspace.knowledge_dir, workspace.competition)
    runnable = sorted(
        p.id for p in plans if is_runnable_plan_status(p.status) and plan_is_selectable(p, retired)
    )
    return runnable[-1] if runnable else None


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
    # Ranked by posterior, not by the prior generation wrote once and never
    # revisited. `confidence` is set at `create()` and updated by nothing, so
    # sorting on it alone kept `hyp:H-010` at 0.99 through the runs that
    # disproved it, and ranked a technique measured as *harmful* above one
    # nobody had tried.
    from labpilot.research_engine.intelligence.hypothesis.selection import rank_hypotheses

    ranked = rank_hypotheses(proposed, workspace.knowledge_dir, workspace.competition)
    return ranked[0].id if ranked else None


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
    # Every micro-agent invocation in this campaign is recorded: which agent,
    # whether the LLM or its rule engine produced the answer, and on failure
    # what kind. M14 2b and 3 are both blocked on having that as *data* rather
    # than log lines, and it can only be collected while the run happens.
    with recording_provenance(
        workspace.knowledge_dir, workspace.competition, session_id=session_id
    ):
        return _run_until_stop_inner(
            store,
            workspace,
            session_id,
            registry,
            llm_client=llm_client,
            max_steps=max_steps,
            auto_approve=auto_approve,
            approval_prompt=approval_prompt,
            on_progress=on_progress,
            autonomy=autonomy,
            campaign_mode=campaign_mode,
            prefer_offline=prefer_offline,
            offline_fallback_prompt=offline_fallback_prompt,
        )


def _run_until_stop_inner(
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
    scheduler = Scheduler(store, registry, workspace, llm_client=llm_client)
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

    # Repair research memory before acting on it. A claim no measurement
    # supports steers every decision this loop is about to make, and the repair
    # must not wait for a *successful experiment* — a campaign that completes
    # none is exactly when memory is most likely to be leading it astray.
    # Measured 2026-08-07: a full campaign ran with 45 false `vit` claims intact
    # because revalidation only fired from `record_successful_execution`.
    try:
        from labpilot.research_engine.evidence.belief_repair import (
            rederive_beliefs_from_cards,
        )
        from labpilot.research_engine.evidence.repair import repair_card_directions
        from labpilot.research_engine.execution.outcome import revalidate_outcome_claims

        # Cards first: revalidation reads their verdicts, so an inverted card
        # would otherwise have its inverted conclusion re-confirmed. Measured
        # 2026-08-07, all 15 rogii cards were built as `maximize=True` on an MSE
        # competition, which recorded the one real improvement as `rejected`.
        # `workspace.root` matters for a workspace whose only `metric.direction`
        # is on the run's own competition.json: without it repair falls back to
        # the knowledge copy and the Analyze profile, and silently no-ops when
        # neither carries a direction.
        reoriented = repair_card_directions(
            workspace.knowledge_dir,
            workspace.competition,
            workspace_root=workspace.root,
        )
        if reoriented:
            _progress(
                f"Re-oriented {len(reoriented)} evidence card(s) built with the "
                "wrong metric direction"
            )

        # Beliefs are recomputed from the repaired cards, not stepped again.
        # Repairing a card does not retract the belief step it already caused,
        # so without this the loop plans against the pre-repair compass.
        rebuilt = rederive_beliefs_from_cards(workspace.knowledge_dir, workspace.competition)
        if rebuilt:
            _progress(f"Re-derived {len(rebuilt)} belief(s) from repaired evidence")

        # Overlays reach the *model*, so a stale one costs more than a stale
        # row. `upsert_skill_overlay` returns early on a known lesson id, so a
        # lesson written from an inverted verdict is permanent until this runs.
        # Measured 2026-08-08: every rogii overlay said `Avoid: SWA` — the only
        # technique that ever improved the metric — long after its card was
        # re-oriented to `accepted`.
        from labpilot.research_engine.evidence.overlay_repair import (
            record_references_in_overlays,
            repair_skill_overlays,
        )

        relearned = repair_skill_overlays(
            workspace.root, workspace.knowledge_dir, workspace.competition
        )
        if relearned:
            _progress(f"Rebuilt {len(relearned)} skill overlay(s) from repaired evidence")
        leaking = record_references_in_overlays(workspace.root)
        if leaking:
            # Not repaired here: the write-path guard in `outcome.py` owns this,
            # and a hit means a write site it does not cover.
            logger.warning("record reference still present in overlay(s): %s", ", ".join(leaking))

        from labpilot.research_engine.execution.technique.vocabulary import (
            recompute_technique_status,
        )

        status_changes = recompute_technique_status(workspace.knowledge_dir, workspace.competition)
        if status_changes:
            _progress(f"Recomputed status for {len(status_changes)} technique(s) from evidence")

        contested = revalidate_outcome_claims(
            knowledge_dir=workspace.knowledge_dir, competition=workspace.competition
        )
        if contested:
            _progress(f"Contested {len(contested)} claim(s) no measurement supports")
    except Exception as exc:  # noqa: BLE001 — never block a campaign on repair
        logger.warning("Claim revalidation at session start failed: %s", exc)

    for step in range(max_steps):
        # Refresh each iteration so mid-session registration is visible.
        allowlist = set(registry.names())
        session = store.get_session(session_id)
        assert session is not None
        if session.status == "paused":
            _progress("Session paused by operator")
            break

        budget_cfg, budget_state = load_budget_pair(session)
        if not submit_tools_allowed(budget_cfg):
            # A campaign told never to submit must not be *offered* the tool.
            # Relying on the approval gate would not hold: `--yes` maps every
            # gated tool to `auto_approve`, so a non-interactive run has no
            # brake between "selected submit_learn" and "uploaded to Kaggle".
            allowlist -= SUBMIT_TOOLS
        stop = evaluate_stops(budget_cfg, budget_state)
        if stop != "none":
            # `failing` is not a completion. Recording it as one would put the
            # campaign that could not run a single experiment in the same state
            # as the campaign that met its target, which is the distinction the
            # breaker exists to draw.
            store.update_session_status(session_id, "failed" if stop == "failing" else "completed")
            if stop != "metric_target":
                store.increment_metric(session_id, "unmet_goal")
            rationale = f"stop:{stop}"
            if stop == "failing":
                # Say what broke. A bare `stop:failing` reproduces the original
                # complaint — a campaign that ended and did not say why.
                why = "; ".join(budget_state.recent_failures[-2:]) or "no successful experiment"
                rationale = (
                    f"stop:{stop} — {budget_state.consecutive_failures} consecutive "
                    f"failed execution(s), {budget_state.steps_since_success} step(s) "
                    f"since the last success. Last: {why}"
                )
            _progress(f"Stop condition: {rationale}")
            decisions.append(
                DecisionRecord(
                    id=store.new_decision_id(),
                    session_id=session_id,
                    tool_name=None,
                    rationale=rationale,
                    stop=True,
                    observe={
                        "stop_reason": stop,
                        "consecutive_failures": budget_state.consecutive_failures,
                        "steps_since_success": budget_state.steps_since_success,
                        "recent_failures": budget_state.recent_failures,
                    },
                )
            )
            store.append_decision(decisions[-1])
            break

        # One step is about to be spent. Counted here rather than on dispatch so
        # a step that never reaches a tool still counts: rogii's S-021 spent 30
        # steps without producing an execution, which no per-execution counter
        # would have noticed.
        budget_state.steps_since_success += 1
        persist_budgets(store, session_id, budget_cfg, budget_state)

        if campaign_mode:
            completed = [
                t.tool_name for t in store.list_tasks(session_id) if t.status == "completed"
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
            # The submit carve-out has to be re-applied: this is the allowlist
            # that reaches `map_research_action`, so a plain re-read would hand
            # the tools back at exactly the point they get selected.
            allowlist = set(registry.names())
            if not submit_tools_allowed(budget_cfg):
                allowlist -= SUBMIT_TOOLS
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
                        f"Policy stopped early with the objective unmet: {research.rationale}",
                        context={"step": step},
                    )
                    # Name the tool explicitly. Routing this by intent text let
                    # keyword matching hijack it: the phrase contains
                    # "hypothesis", which matches the ("plan", "baseline",
                    # "hypothesis") template before anything else, so the
                    # override dispatched generate_plan(baseline=True) instead
                    # of reflecting on the result it was reacting to.
                    override_tool = "reflect" if "reflect" in allowlist else "generate_plan"
                    research = ResearchAction(
                        intent=f"objective unmet — {override_tool} and continue",
                        rationale="objective still unmet; continuing",
                        suggested_tools=[override_tool],
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
                        store.update_task_status(task.id, "cancelled", error="operator rejected")
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
                    if tool_step.tool in _EXPERIMENT_TOOLS:
                        # Ask what the *execution* did, not whether the call
                        # returned. `run_plan` reports a failed execution in its
                        # result and raises nothing, so counting a clean return
                        # as a success reset the breaker on every one.
                        #
                        # Measured 2026-08-09: a campaign with **8 executions,
                        # all failed** ran its full 8 steps, because 10 of its
                        # 16 dispatches were `run_plan` returning normally. The
                        # breaker built to stop exactly that never reached 3.
                        outcome = _experiment_outcome(result)
                        _record_experiment_outcome(
                            store,
                            session_id,
                            succeeded=outcome[0],
                            error=outcome[1],
                        )
                except LLMDegradedError as exc:
                    # Strict mode (M14 2b) means fatal, and the generic handler
                    # below would reduce it to one lost step with the campaign
                    # carrying on — which is the silent degradation M14 exists
                    # to remove, just one level up. Stop the session and say why.
                    _fail_session_on_degraded_llm(store, session_id, record, decisions, exc)
                    _progress(f"Campaign stopped: {exc}")
                    raise
                except Exception as exc:
                    store.increment_metric(session_id, "tasks_failed")
                    if tool_step.tool in _EXPERIMENT_TOOLS:
                        _record_experiment_outcome(
                            store, session_id, succeeded=False, error=str(exc)
                        )
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
        # Resolve `@latest` here too. Only the multi-step campaign path did,
        # so this path could not use the shared `_default_args` and hand-rolled
        # its own — which is how `offline_next_action` came to pin
        # `plan_id="P-001"` and `dry_run=True`, and how a degraded policy
        # started minting `dry_run_stub` metrics that became evidence cards.
        action_args = resolve_step_args(
            action.tool,
            action.args,
            latest_plan_id=_latest_plan_id(workspace),
            latest_execution_id=_latest_execution_id(workspace),
            next_hypothesis_id=_next_hypothesis_id(workspace),
            baseline_plan_exists=_baseline_plan_exists(workspace),
        )
        record.args = action_args
        task = store.enqueue(
            session_id,
            action.tool,
            args=action_args,
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
        except LLMDegradedError as exc:
            _fail_session_on_degraded_llm(store, session_id, record, decisions, exc)
            _progress(f"Campaign stopped: {exc}")
            raise
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
