"""Shared helpers used by every pillar (ids, JSON-in-TEXT, Micro Agent contract).

Import rule: ``accessor.common`` may use stdlib / third-party and other
``accessor`` modules only — never ``research_engine`` or ``cli``.
"""

from labpilot.accessor.common.atomic_write import atomic_write_text
from labpilot.accessor.common.file_lock import locked
from labpilot.accessor.common.ids import allocate_sequential_id, task_id

__all__ = ["allocate_sequential_id", "atomic_write_text", "locked", "task_id"]
