"""Campaign checkpoint helpers — persist restoreable session campaign state."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    budgets_from_metadata,
    budgets_to_metadata,
)
from labpilot.research_engine.conductor.models import ConductSession
from labpilot.research_engine.conductor.store import ConductorStore


def save_checkpoint(
    store: ConductorStore,
    session_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge campaign checkpoint fields into session metadata."""
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    meta = dict(session.metadata)
    tasks = store.list_tasks(session_id)
    decisions = store.list_decisions(session_id)
    meta["checkpoint"] = {
        "task_count": len(tasks),
        "decision_count": len(decisions),
        "pending": sum(1 for t in tasks if t.status == "pending"),
        "completed_tools": [t.tool_name for t in tasks if t.status == "completed"],
        "last_decision_id": decisions[-1].id if decisions else None,
        **(extra or {}),
    }
    store.update_session_metadata(session_id, meta)
    return meta["checkpoint"]


def load_budget_pair(
    session: ConductSession,
) -> tuple[BudgetConfig, BudgetState]:
    return budgets_from_metadata(session.metadata)


def persist_budgets(
    store: ConductorStore,
    session_id: str,
    config: BudgetConfig,
    state: BudgetState,
) -> None:
    session = store.get_session(session_id)
    if session is None:
        raise ValueError(f"unknown session: {session_id}")
    meta = budgets_to_metadata(session.metadata, config, state)
    store.update_session_metadata(session_id, meta)


def latest_active_session(store: ConductorStore) -> ConductSession | None:
    """Most recently updated non-terminal session for this competition."""
    sessions = store.list_sessions()
    active = [s for s in sessions if s.status in {"running", "paused", "waiting"}]
    if not active:
        return None
    active.sort(key=lambda s: s.updated_at, reverse=True)
    return active[0]
