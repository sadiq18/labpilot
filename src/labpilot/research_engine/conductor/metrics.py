"""Campaign metrics and capability-gap suggestions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.conductor.models import _now
from labpilot.research_engine.conductor.store import ConductorStore


class CampaignMetrics(BaseModel):
    session_id: str
    tasks_failed: int = 0
    tasks_blocked: int = 0
    unmet_goal: int = 0
    human_interventions: int = 0
    no_capability: int = 0
    submissions: int = 0
    llm_cost_usd: float = 0.0
    updated_at: str = Field(default_factory=_now)


class Suggestion(BaseModel):
    id: str
    session_id: str
    kind: str = "no_capability"
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)


def ensure_metrics(store: ConductorStore, session_id: str) -> CampaignMetrics:
    existing = store.get_metrics(session_id)
    if existing is not None:
        return existing
    metrics = CampaignMetrics(session_id=session_id)
    store.upsert_metrics(metrics)
    return metrics


def record_suggestion(
    store: ConductorStore,
    session_id: str,
    message: str,
    *,
    kind: str = "no_capability",
    context: dict[str, Any] | None = None,
) -> Suggestion:
    ensure_metrics(store, session_id)
    store.increment_metric(session_id, "no_capability")
    suggestion = Suggestion(
        id=store.new_suggestion_id(),
        session_id=session_id,
        kind=kind,
        message=message,
        context=context or {},
    )
    store.append_suggestion(suggestion)
    return suggestion
