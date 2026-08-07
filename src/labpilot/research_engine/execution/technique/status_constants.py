"""Vocabulary status constants — no imports from intelligence/retrieval."""

from __future__ import annotations

VALID_STATUSES = frozenset({"candidate", "confirmed", "rejected", "dormant"})
PLANNER_VISIBLE_STATUSES = frozenset({"candidate", "confirmed"})
CLAIM_PROMOTION_STATUSES = frozenset({"confirmed"})
