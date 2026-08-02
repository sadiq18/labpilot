"""Capability gap ledger — aggregate ``no_capability`` suggestions.

Local SQLite rollup for debugging and maintainer review. Product-wide visibility
still requires telemetry/export before public launch (see backlog).
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.conductor.models import _now
from labpilot.research_engine.conductor.store import ConductorStore

DecisionKind = Literal["promote", "alias", "defer", "reject"]

_NEED_TOOL_RE = re.compile(
    r"Need capability/tool ['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

_SAMPLE_CAP = 5
_MAINTAINER_TRUTHY = frozenset({"1", "true", "yes", "on"})


class CapabilityGap(BaseModel):
    gap_key: str
    kind: str = "no_capability"
    count: int = 0
    first_seen_at: str = ""
    last_seen_at: str = ""
    sample_contexts: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "open"
    promoted_tool: str | None = None
    decision_reason: str = ""
    decided_at: str | None = None


class CapabilityDecision(BaseModel):
    id: str
    gap_key: str
    decision: DecisionKind
    reason: str = ""
    promoted_tool: str | None = None
    created_at: str = Field(default_factory=_now)


def is_maintainer_enabled(
    env: dict[str, str] | None = None,
) -> bool:
    """True when ``LABPILOT_MAINTAINER`` authorizes promote/defer/reject."""
    source = env if env is not None else os.environ
    return source.get("LABPILOT_MAINTAINER", "").strip().lower() in _MAINTAINER_TRUTHY


def normalize_gap_key(
    *,
    missing_tools: list[str] | None = None,
    intent: str = "",
    message: str = "",
) -> str:
    """Stable key: prefer missing tool name, else intent, else message hash."""
    tools = [t.strip().lower() for t in (missing_tools or []) if t and t.strip()]
    if tools:
        return f"tool:{tools[0]}"
    match = _NEED_TOOL_RE.search(message or "")
    if match:
        return f"tool:{match.group(1).strip().lower()}"
    intent_n = " ".join((intent or "").lower().split())
    if intent_n:
        return f"intent:{intent_n[:120]}"
    digest = hashlib.sha256((message or "").encode("utf-8")).hexdigest()[:12]
    return f"msg:{digest}"


def build_suggestion_context(
    *,
    intent: str,
    suggested_tools: list[str] | None = None,
    missing_tools: list[str] | None = None,
    competition: str = "",
    session_id: str = "",
    goal: str = "",
) -> dict[str, Any]:
    """Structured context for ``os_suggestions`` / gap samples."""
    ctx: dict[str, Any] = {"intent": intent}
    if suggested_tools:
        ctx["suggested_tools"] = list(suggested_tools)
    if missing_tools:
        ctx["missing_tools"] = list(missing_tools)
    if competition:
        ctx["competition"] = competition
    if session_id:
        ctx["session_id"] = session_id
    if goal:
        # Stored locally for debugging; stripped from redacted export.
        ctx["goal"] = goal
    return ctx


def redact_context(context: dict[str, Any]) -> dict[str, Any]:
    """Drop goal / session identifiers from exported samples."""
    keep_keys = ("intent", "missing_tools", "suggested_tools", "competition")
    return {k: context[k] for k in keep_keys if k in context and context[k]}


def note_suggestion(store: ConductorStore, suggestion: Any) -> CapabilityGap:
    """Upsert gap ledger from a persisted Suggestion."""
    context = dict(getattr(suggestion, "context", None) or {})
    missing = list(context.get("missing_tools") or [])
    intent = str(context.get("intent") or "")
    message = str(getattr(suggestion, "message", "") or "")
    kind = str(getattr(suggestion, "kind", None) or "no_capability")
    gap_key = normalize_gap_key(
        missing_tools=missing,
        intent=intent,
        message=message,
    )
    sample = redact_context(context)
    if not sample and intent:
        sample = {"intent": intent}
    return store.upsert_capability_gap(
        gap_key,
        kind=kind,
        sample_context=sample,
    )


def export_gaps_payload(
    store: ConductorStore,
    *,
    status: str | None = "open",
    competition: str = "",
) -> dict[str, Any]:
    """Redacted aggregate suitable for maintainer review / future telemetry."""
    gaps = store.list_capability_gaps(status=status)
    return {
        "schema": "labpilot.capability_gaps/v1",
        "competition": competition or store.competition,
        "exported_at": _now(),
        "gaps": [
            {
                "gap_key": g.gap_key,
                "kind": g.kind,
                "count": g.count,
                "first_seen_at": g.first_seen_at,
                "last_seen_at": g.last_seen_at,
                "status": g.status,
                "promoted_tool": g.promoted_tool,
                "sample_contexts": [redact_context(c) for c in g.sample_contexts],
            }
            for g in gaps
        ],
    }


def apply_gap_decision(
    store: ConductorStore,
    gap_key: str,
    decision: DecisionKind,
    *,
    reason: str = "",
    promoted_tool: str | None = None,
    require_maintainer: bool = True,
    env: dict[str, str] | None = None,
) -> CapabilityDecision:
    """Record promote/alias/defer/reject on a local gap (maintainer-gated)."""
    if require_maintainer and not is_maintainer_enabled(env):
        raise PermissionError(
            "Gap decisions require LABPILOT_MAINTAINER=1 "
            "(end users cannot promote into the shared catalog)"
        )
    if decision in {"promote", "alias"} and not (promoted_tool or "").strip():
        raise ValueError(f"{decision} requires promoted_tool")
    gap = store.get_capability_gap(gap_key)
    if gap is None:
        raise KeyError(f"unknown gap_key: {gap_key}")
    status_map: dict[DecisionKind, str] = {
        "promote": "promoted",
        "alias": "alias",
        "defer": "deferred",
        "reject": "rejected",
    }
    tool = (promoted_tool or "").strip() or None
    store.update_capability_gap_status(
        gap_key,
        status=status_map[decision],
        promoted_tool=tool,
        decision_reason=reason,
    )
    record = CapabilityDecision(
        id=store.new_capability_decision_id(),
        gap_key=gap_key,
        decision=decision,
        reason=reason,
        promoted_tool=tool,
    )
    store.append_capability_decision(record)
    return record
