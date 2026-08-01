"""Metadata filters for ContextItem candidates."""

from __future__ import annotations

from labpilot.research_engine.context.models import ContextItem, ContextRequest


def apply_filters(
    items: list[ContextItem],
    request: ContextRequest,
) -> list[ContextItem]:
    """Drop items that fail competition / kind / status filters."""
    kinds = request.kinds
    statuses = request.statuses
    out: list[ContextItem] = []
    for item in items:
        meta = item.metadata or {}
        item_comp = meta.get("competition")
        if (
            request.filter_competition
            and item_comp is not None
            and str(item_comp) != request.competition
        ):
            continue
        if kinds is not None and item.kind not in kinds:
            continue
        if statuses is not None:
            item_status = meta.get("status")
            if item_status not in statuses:
                continue
        out.append(item)
    return out
