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

    A missing status defaults to `"candidate"` so the vocabulary can still
    grow — every caller filtering a technique by status needs this same
    default, and four call sites (`filter_by_technique_status`, the
    stagnation minter's `_untried_technique`, `_select_techniques` in
    `retrieval/fetchers.py`, and `generate_candidates`) had drifted into
    reimplementing it separately. Checked against `status is None`, not
    truthiness: `dict.get(key, "candidate")`, the pattern this replaces,
    only applies the default when the *key* is absent — an explicit empty
    status stored against a present key stays empty (and so is correctly
    filtered out) rather than being treated as `"candidate"`.
    """
    resolved = status if status is not None else "candidate"
    return resolved in visible


#: Claims never promote rejected/dormant; measurement remains the confirmed bar.
CLAIM_BLOCKED_STATUSES = frozenset({"rejected", "dormant"})

#: Unmeasured + never selected becomes dormant after this many later campaigns.
DORMANT_AFTER_CAMPAIGNS = 2
