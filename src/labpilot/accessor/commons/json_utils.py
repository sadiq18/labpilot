"""JSON-in-TEXT helpers for the SQLite convention (list/dict columns as TEXT)."""

from __future__ import annotations

import json
from typing import Any


def dumps(value: Any) -> str:
    """Serialize a Python value for a TEXT column."""
    return json.dumps(value, ensure_ascii=False)


def loads(text: str | None, default: Any = None) -> Any:
    """Deserialize a TEXT column, returning ``default`` on empty/invalid input."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default
