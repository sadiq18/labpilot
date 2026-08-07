"""Technique vocabulary status labels (M-25).

Kept free of upstream imports so store/vocabulary can share them without cycles.
"""

from __future__ import annotations

VALID_STATUSES = frozenset({"candidate", "confirmed", "rejected", "dormant"})
