"""Retrieve pipeline helpers — filter then BM25-score candidates."""

from __future__ import annotations

from labpilot.research_engine.context.bm25 import bm25_scores
from labpilot.research_engine.context.filters import apply_filters
from labpilot.research_engine.context.models import ContextItem, ContextRequest


def retrieve_candidates(
    items: list[ContextItem],
    request: ContextRequest,
) -> list[ContextItem]:
    """Filter by metadata, score with BM25, sort descending, apply max_items."""
    filtered = apply_filters(items, request)
    query = (request.query or request.goal or "").strip()
    if not filtered:
        return []

    if query:
        texts = [f"{item.kind} {item.text} {item.reason}" for item in filtered]
        scores = bm25_scores(texts, query)
        scored: list[ContextItem] = []
        for item, score in zip(filtered, scores, strict=True):
            combined = float(score) if score > 0 else float(item.score) * 0.01
            scored.append(
                item.model_copy(
                    update={
                        "score": combined,
                        "reason": (
                            f"{item.reason} | bm25={score:.4f}".strip(" |")
                            if item.reason
                            else f"bm25={score:.4f}"
                        ),
                    }
                )
            )
        scored.sort(key=lambda i: i.score, reverse=True)
        filtered = scored
    else:
        filtered = sorted(filtered, key=lambda i: i.score, reverse=True)

    if request.max_items >= 0:
        filtered = filtered[: request.max_items]
    return filtered
