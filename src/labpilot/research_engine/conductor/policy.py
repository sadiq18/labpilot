"""Constrained Conductor policy — NextAction from allowlisted tools only."""

from __future__ import annotations

import json
import logging
from typing import Any

from labpilot.research_engine.conductor.approvals import (
    OfflineFallbackPrompt,
    resolve_offline_fallback,
)
from labpilot.research_engine.conductor.models import NextAction
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)

# Offline / null-LLM fallback order (deterministic).
_DEFAULT_ORDER = (
    "analyze_competition",
    "search_papers",
    "query_memory",
    "generate_plan",
    "run_plan",
    "reflect",
    "submit",
)


def build_observe_bundle(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    *,
    include_context: bool = True,
    max_context_items: int = 16,
    max_context_chars: int = 4000,
) -> dict[str, Any]:
    """Gather durable state for policy input.

    When ``include_context`` is true (online path), attach a best-effort
    Context Engine summary and ranked refs. Failures never raise — observe
    always remains usable for offline / LLM policy.
    """
    session = store.get_session(session_id)
    tasks = store.list_tasks(session_id)
    feedback = store.list_feedback(session_id, limit=10)
    decisions = store.list_decisions(session_id)
    observe: dict[str, Any] = {
        "competition": workspace.competition,
        "goal": session.goal if session else "",
        "session_status": session.status if session else None,
        "layout": workspace.layout,
        "task_summary": [
            {
                "id": t.id,
                "tool": t.tool_name,
                "status": t.status,
                "error": t.error,
            }
            for t in tasks
        ],
        "completed_tools": [t.tool_name for t in tasks if t.status == "completed"],
        "operator_feedback": [
            {
                "gated_tool": f.gated_tool,
                "decision": f.decision,
                "comment": f.comment,
            }
            for f in feedback
        ],
        "recent_rationales": [
            {"tool": d.tool_name, "rationale": d.rationale, "stop": d.stop}
            for d in decisions[-5:]
        ],
    }
    _attach_evidence_refresh(observe, workspace)
    if include_context:
        _attach_context(
            observe,
            workspace,
            session_id,
            max_items=max_context_items,
            max_chars=max_context_chars,
        )
    return observe


def _attach_evidence_refresh(observe: dict[str, Any], workspace: Workspace) -> None:
    """Surface bus-written evidence refresh notes for policy (best-effort)."""
    note = workspace.root / "artifacts" / f"evidence_refresh_{workspace.competition}.json"
    if not note.is_file():
        return
    try:
        data = json.loads(note.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict):
        observe["evidence_refresh"] = data


def _attach_context(
    observe: dict[str, Any],
    workspace: Workspace,
    session_id: str,
    *,
    max_items: int = 16,
    max_chars: int = 4000,
) -> None:
    """Best-effort Context Engine attach; never raises. Mutates ``observe``."""
    try:
        from labpilot.research_engine.context import ContextRequest, build_context

        goal = str(observe.get("goal") or "")
        request = ContextRequest(
            competition=workspace.competition,
            goal=goal,
            query=goal,
            session_id=session_id,
            knowledge_dir=workspace.knowledge_dir,
            max_items=max_items,
            max_chars=max_chars,
        )
        bundle = build_context(request)
        observe["context_summary"] = bundle.summary(max_chars=2000)
        observe["context_refs"] = [
            {
                "id": item.id,
                "source": item.source,
                "kind": item.kind,
                "score": item.score,
                "reason": item.reason,
            }
            for item in bundle.items
        ]
        if bundle.provider_errors:
            observe["context_provider_errors"] = list(bundle.provider_errors)
    except Exception as exc:  # noqa: BLE001 — observe must stay usable
        logger.warning("Context Engine unavailable for observe: %s", exc)
        observe.setdefault("context_summary", "")
        observe.setdefault("context_refs", [])
        observe["context_provider_errors"] = [f"build_context: {exc}"]


def offline_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
) -> NextAction:
    """Deterministic next tool: first catalog tool not yet completed."""
    done = set(observe.get("completed_tools") or [])
    # Honours reject-submit feedback: skip submit if last feedback rejected it.
    feedback = observe.get("operator_feedback") or []
    rejected = {
        f["gated_tool"]
        for f in feedback
        if f.get("decision") == "reject"
    }
    for name in _DEFAULT_ORDER:
        if name not in allowlist:
            continue
        if name in done:
            continue
        if name in rejected:
            continue
        args: dict[str, Any] = {}
        if name == "generate_plan":
            args = {"baseline": True}
        if name == "search_papers":
            args = {"offline": True}
        if name == "run_plan":
            # Caller/loop may inject plan_id; offline stub uses placeholder.
            args = {"plan_id": "P-001", "dry_run": True}
        return NextAction(
            tool=name,
            args=args,
            rationale=f"offline policy: next unfinished tool is {name}",
            stop=False,
        )
    return NextAction(tool=None, rationale="offline policy: catalog exhausted", stop=True)


def validate_next_action(action: NextAction, allowlist: set[str]) -> NextAction:
    """Reject invented tools; force stop if invalid."""
    if action.stop or not action.tool:
        return NextAction(
            tool=None,
            args={},
            rationale=action.rationale or "stop",
            stop=True,
        )
    if action.tool not in allowlist:
        return NextAction(
            tool=None,
            args={},
            rationale=f"rejected non-catalog tool: {action.tool}",
            stop=True,
        )
    return action


def _invoke_llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any,
) -> NextAction:
    catalog = sorted(allowlist)
    system = (
        "You are the LabPilot Research Conductor. Choose the single next tool "
        "from the allowlist, or stop. Never invent tools. Prefer operator_feedback "
        "comments when deciding. Use context_summary and context_refs as ranked "
        "evidence (higher score is stronger). Respond with JSON only: "
        '{"tool": "<name>|null", "args": {}, "rationale": "...", "stop": false}'
    )
    user = json.dumps(
        {"allowlist": catalog, "observe": observe},
        indent=2,
        default=str,
    )
    if hasattr(llm_client, "complete"):
        text = llm_client.complete(system, user)
    elif hasattr(llm_client, "generate"):
        text = str(llm_client.generate(task="planning", prompt=user))
    else:
        raise TypeError("llm_client has no complete/generate method")
    data = _parse_json(text)
    action = NextAction.model_validate(data)
    return validate_next_action(action, allowlist)


def llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any | None,
    *,
    prefer_offline: bool = False,
    auto_offline_fallback: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
    max_llm_retries: int = 5,
) -> NextAction:
    """Ask the LLM for a structured NextAction.

    On LLM failure (or missing client in online mode), ask the operator before
    using the deterministic offline order: allow, deny, or retry.
    Intentional ``prefer_offline`` skips the prompt.
    """
    if prefer_offline:
        return offline_next_action(observe, allowlist)

    retries = 0
    while True:
        if llm_client is None:
            reason = "No LLM client available"
        else:
            try:
                return _invoke_llm_next_action(observe, allowlist, llm_client)
            except Exception as exc:
                reason = f"LLM policy failed: {exc}"
                logger.warning("Conductor policy LLM failed (%s)", exc)

        decision = resolve_offline_fallback(
            reason,
            auto=auto_offline_fallback,
            prompt=offline_fallback_prompt,
        )
        if decision == "allow":
            logger.info("Operator allowed offline policy fallback (%s)", reason)
            return offline_next_action(observe, allowlist)
        if decision == "deny":
            logger.info("Operator denied offline policy fallback (%s)", reason)
            return NextAction(
                tool=None,
                rationale=f"operator denied offline policy fallback ({reason})",
                stop=True,
            )
        # retry
        retries += 1
        if retries > max_llm_retries:
            logger.warning(
                "Exceeded max LLM retries (%s); treating as deny", max_llm_retries
            )
            return NextAction(
                tool=None,
                rationale=(
                    f"operator retry exhausted after {max_llm_retries} attempts "
                    f"({reason})"
                ),
                stop=True,
            )
        logger.info("Operator requested LLM policy retry (%d/%d)", retries, max_llm_retries)


def available_tools(workspace: Workspace, allowlist: set[str]) -> set[str]:
    """Drop tools whose preconditions the workspace does not yet satisfy.

    Offering the whole catalog regardless of state lets a campaign burn steps
    on impossible work — reflecting before anything has run, or submitting
    before a model exists. Filtering first turns "the model picked badly" into
    "that option was never on the table".
    """
    from labpilot.research_engine.conductor.loop import (
        _latest_execution_id,
        _latest_plan_id,
    )

    has_plan = _latest_plan_id(workspace) is not None
    has_execution = _latest_execution_id(workspace) is not None

    requires: dict[str, bool] = {
        # Nothing to reflect on until an experiment has produced evidence.
        "reflect": has_execution,
        # Cannot run, or submit the result of, a plan that does not exist.
        "run_plan": has_plan,
        "run_experiment": has_plan,
        "submit": has_execution,
        "submit_learn": has_execution,
    }
    return {name for name in allowlist if requires.get(name, True)}


def decide_next(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
    prefer_offline: bool = False,
    auto_offline_fallback: bool = False,
    offline_fallback_prompt: OfflineFallbackPrompt | None = None,
) -> tuple[NextAction, dict[str, Any]]:
    """Observe + think; return validated NextAction and observe bundle.

    Online path attaches Context Engine evidence to observe. ``prefer_offline``
    skips retrieve entirely (no forced Context Engine success).
    """
    allowlist = available_tools(workspace, set(registry.names()))
    observe = build_observe_bundle(
        store,
        workspace,
        session_id,
        include_context=not prefer_offline,
    )
    action = llm_next_action(
        observe,
        allowlist,
        llm_client,
        prefer_offline=prefer_offline,
        auto_offline_fallback=auto_offline_fallback,
        offline_fallback_prompt=offline_fallback_prompt,
    )
    return action, observe


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data
