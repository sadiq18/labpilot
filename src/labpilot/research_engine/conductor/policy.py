"""Constrained Conductor policy — NextAction from allowlisted tools only."""

from __future__ import annotations

import json
import logging
from typing import Any

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
) -> dict[str, Any]:
    """Gather durable state for policy input (no full context engine)."""
    session = store.get_session(session_id)
    tasks = store.list_tasks(session_id)
    feedback = store.list_feedback(session_id, limit=10)
    decisions = store.list_decisions(session_id)
    return {
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


def llm_next_action(
    observe: dict[str, Any],
    allowlist: set[str],
    llm_client: Any | None,
) -> NextAction:
    """Ask the LLM for a structured NextAction; fall back to offline on failure."""
    if llm_client is None:
        return offline_next_action(observe, allowlist)

    catalog = sorted(allowlist)
    system = (
        "You are the LabPilot Research Conductor. Choose the single next tool "
        "from the allowlist, or stop. Never invent tools. Prefer operator_feedback "
        "comments when deciding. Respond with JSON only: "
        '{"tool": "<name>|null", "args": {}, "rationale": "...", "stop": false}'
    )
    user = json.dumps(
        {"allowlist": catalog, "observe": observe},
        indent=2,
        default=str,
    )
    try:
        if hasattr(llm_client, "complete"):
            text = llm_client.complete(system, user)
        elif hasattr(llm_client, "generate"):
            text = str(llm_client.generate(task="planning", prompt=user))
        else:
            return offline_next_action(observe, allowlist)
        data = _parse_json(text)
        action = NextAction.model_validate(data)
        return validate_next_action(action, allowlist)
    except Exception as exc:
        logger.warning("Conductor policy LLM failed (%s); using offline fallback", exc)
        return offline_next_action(observe, allowlist)


def decide_next(
    store: ConductorStore,
    workspace: Workspace,
    session_id: str,
    registry: ToolRegistry,
    *,
    llm_client: Any | None = None,
) -> tuple[NextAction, dict[str, Any]]:
    """Observe + think; return validated NextAction and observe bundle."""
    allowlist = set(registry.names())
    observe = build_observe_bundle(store, workspace, session_id)
    action = llm_next_action(observe, allowlist, llm_client)
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
