"""RI retrieval provider — adapts existing ContextBuilder into ContextItems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio

from labpilot.research_engine.context.models import ContextItem, ContextRequest
from labpilot.research_engine.intelligence.retrieval.context_builder import (
    build_research_context,
)
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext


class RIRetrievalProvider:
    """Adapter: existing RI ``ResearchContext`` → ``ContextItem`` list."""

    name = "ri_retrieval"

    def __init__(self, *, llm_client: Any | None = None) -> None:
        self.llm_client = llm_client

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        if request.knowledge_dir is None:
            return []
        return await anyio.to_thread.run_sync(self._fetch_sync, request)

    def _fetch_sync(self, request: ContextRequest) -> list[ContextItem]:
        knowledge_dir = Path(request.knowledge_dir)
        query = (request.query or request.goal or "").strip()
        ctx = build_research_context(
            knowledge_dir,
            request.competition,
            query,
            llm_client=self.llm_client,
        )
        items = research_context_to_items(ctx, source=self.name)
        stamped: list[ContextItem] = []
        for item in items:
            meta = dict(item.metadata)
            meta.setdefault("competition", request.competition)
            stamped.append(item.model_copy(update={"metadata": meta}))
        return stamped


def research_context_to_items(
    ctx: ResearchContext,
    *,
    source: str = "ri_retrieval",
) -> list[ContextItem]:
    """Flatten typed ResearchContext fields into ContextItems."""
    items: list[ContextItem] = []

    if ctx.brief:
        items.append(
            ContextItem(
                id=f"{source}:brief",
                source=source,
                kind="brief",
                text=ctx.brief,
                score=1.0,
                reason="RI compressed brief",
            )
        )

    for i, tech in enumerate(ctx.techniques):
        name = str(tech.get("name") or tech.get("id") or f"technique-{i}")
        text = tech.get("render") or tech.get("benefits") or name
        if callable(text):
            text = name
        items.append(
            ContextItem(
                id=f"{source}:technique:{tech.get('id', i)}",
                source=source,
                kind="technique",
                text=str(text),
                score=float(tech.get("confidence") or 0.5),
                reason="RI technique card",
                metadata={"raw": tech},
            )
        )

    for kind, rows in (
        ("paper", ctx.papers),
        ("experiment", ctx.experiments),
        ("repository", ctx.repositories),
        ("failure", ctx.failures),
    ):
        for i, row in enumerate(rows):
            label = str(row.get("label") or row.get("name") or row.get("id") or f"{kind}-{i}")
            summary = str(row.get("summary") or row.get("why") or label)
            items.append(
                ContextItem(
                    id=f"{source}:{kind}:{row.get('document_id') or row.get('id') or i}",
                    source=source,
                    kind=kind,
                    text=summary,
                    score=float(row.get("score") or 0.5),
                    reason=str(row.get("why") or f"RI {kind}"),
                    metadata={"raw": row},
                )
            )

    for i, note in enumerate(ctx.notes):
        items.append(
            ContextItem(
                id=f"{source}:note:{i}",
                source=source,
                kind="note",
                text=str(note),
                score=0.3,
                reason="RI note",
            )
        )

    return items
