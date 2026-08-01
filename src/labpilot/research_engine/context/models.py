"""Context Engine models — request + durable ContextBundle."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ContextRequest(BaseModel):
    """Inputs for ``build_context`` — goal/task oriented, storage-agnostic."""

    competition: str
    goal: str = ""
    query: str = ""
    session_id: str | None = None
    task_id: str | None = None
    knowledge_dir: Path | None = None
    max_items: int = 32
    kinds: list[str] | None = None
    statuses: list[str] | None = None
    filter_competition: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextItem(BaseModel):
    """One evidence snippet from a provider (pre- or post-rank)."""

    id: str
    source: str
    kind: str
    text: str
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBundle(BaseModel):
    """Prompt-ready context for Conductor / CLI / agents."""

    request: ContextRequest
    items: list[ContextItem] = Field(default_factory=list)
    provider_errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    graph_metrics: GraphQueryMetrics = Field(default_factory=GraphQueryMetrics)
    built_at: str = Field(default_factory=_now)

    def summary(self, *, max_chars: int = 2000) -> str:
        """Compact text view for observe / CLI."""
        parts: list[str] = []
        for item in self.items:
            line = f"[{item.source}/{item.kind}] {item.text}".strip()
            parts.append(line)
        text = "\n".join(parts)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text
