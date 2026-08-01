"""Optional event hooks — pub/sub lands later; specialists emit no-ops for now."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

EventEmitter = Callable[[str, dict[str, Any]], None]


def noop_emit(event: str, payload: dict[str, Any]) -> None:
    """Discard specialist lifecycle events (bus subscribers attach later)."""
    del event, payload
