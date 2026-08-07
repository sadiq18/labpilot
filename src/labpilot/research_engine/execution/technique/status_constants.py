"""Technique vocabulary status labels (M-25).

Kept free of upstream imports so store/vocabulary/consumers can share them
without import cycles.
"""

from __future__ import annotations

VALID_STATUSES = frozenset({"candidate", "confirmed", "rejected", "dormant"})

#: Planner / retrieval / candidate generation.
PLANNER_VISIBLE_STATUSES = frozenset({"candidate", "confirmed"})

#: Claims never promote rejected/dormant; measurement remains the confirmed bar.
CLAIM_BLOCKED_STATUSES = frozenset({"rejected", "dormant"})

#: Unmeasured + never selected becomes dormant after this many later campaigns.
DORMANT_AFTER_CAMPAIGNS = 2
