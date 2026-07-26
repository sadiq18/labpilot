"""Shared low-level helpers (id allocation, JSON-in-TEXT) — no schema here."""

from labpilot.accessor.commons.ids import allocate_sequential_id, task_id

__all__ = ["allocate_sequential_id", "task_id"]
