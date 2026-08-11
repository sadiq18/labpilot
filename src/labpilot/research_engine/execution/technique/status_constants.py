"""Technique vocabulary status labels (M-25).

Kept free of upstream imports so store/vocabulary/consumers can share them
without import cycles.
"""

from __future__ import annotations

VALID_STATUSES = frozenset({"candidate", "confirmed", "rejected", "dormant"})

#: Planner / retrieval / candidate generation.
PLANNER_VISIBLE_STATUSES = frozenset({"candidate", "confirmed"})


def is_planner_visible(
    status: str | None, *, visible: frozenset[str] = PLANNER_VISIBLE_STATUSES
) -> bool:
    """Whether a technique's vocabulary status may reach the planner.

    Unknown names default to `"candidate"` so the vocabulary can still grow —
    every caller filtering a technique by status needs this same default, and
    the two that existed before this helper (`filter_by_technique_status`,
    the stagnation minter's `_untried_technique`) had drifted into
    reimplementing it separately.
    """
    return str(status or "candidate") in visible


#: Claims never promote rejected/dormant; measurement remains the confirmed bar.
CLAIM_BLOCKED_STATUSES = frozenset({"rejected", "dormant"})

#: Unmeasured + never selected becomes dormant after this many later campaigns.
DORMANT_AFTER_CAMPAIGNS = 2
