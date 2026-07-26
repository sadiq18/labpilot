"""Deterministic id allocators shared across stores.

Mirrors the ``H-001`` style used by ``HypothesisStore``: scan existing ids for
the highest integer suffix and return the next one, zero-padded.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def allocate_sequential_id(
    prefix: str,
    existing: Iterable[str],
    *,
    sep: str = "-",
    width: int = 3,
) -> str:
    """Return the next ``<prefix><sep><n>`` id after the highest in ``existing``.

    Example: ``allocate_sequential_id("P", ["P-001", "P-002"])`` -> ``"P-003"``.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(sep)}0*(\d+)$")
    max_n = 0
    for value in existing:
        match = pattern.match(str(value))
        if match:
            max_n = max(max_n, int(match.group(1)))
    next_n = max_n + 1
    pad = max(width, len(str(next_n)))
    return f"{prefix}{sep}{next_n:0{pad}d}"


def task_id(plan_id: str, index: int, *, width: int = 2) -> str:
    """Task id under a plan, e.g. ``task_id("P-001", 1)`` -> ``"P-001-T01"``."""
    pad = max(width, len(str(index)))
    return f"{plan_id}-T{index:0{pad}d}"
