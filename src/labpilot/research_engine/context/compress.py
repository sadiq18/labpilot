"""Compress ranked candidates into a budgeted ContextBundle payload.

Budgets (from ``ContextRequest``)::

    max_items       — hard cap on number of items
    max_chars       — total characters across kept item texts
    max_item_chars  — truncate each item text before packing
"""

from __future__ import annotations

from labpilot.research_engine.context.models import ContextItem, ContextRequest


def compress_candidates(
    items: list[ContextItem],
    request: ContextRequest,
) -> list[ContextItem]:
    """Pack highest-ranked items under item-count and character budgets."""
    max_items = request.max_items
    max_chars = max(0, int(request.max_chars))
    max_item_chars = max(0, int(request.max_item_chars))

    kept: list[ContextItem] = []
    used_chars = 0

    for item in items:
        if max_items >= 0 and len(kept) >= max_items:
            break
        text = item.text or ""
        truncated = False
        if max_item_chars > 0 and len(text) > max_item_chars:
            text = text[: max_item_chars - 3] + "..."
            truncated = True
        if max_chars > 0 and used_chars + len(text) > max_chars:
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if remaining < 16 and kept:
                # Too little room for a useful snippet — stop.
                break
            text = text[: max(0, remaining - 3)] + ("..." if remaining > 3 else "")
            truncated = True
            if not text.strip():
                break

        reason = item.reason
        if truncated:
            reason = f"{reason} | compressed".strip(" |")
        kept.append(
            item.model_copy(
                update={
                    "text": text,
                    "reason": reason,
                    "metadata": {
                        **item.metadata,
                        "compressed": truncated,
                    },
                }
            )
        )
        used_chars += len(text)
        if max_chars > 0 and used_chars >= max_chars:
            break

    return kept
