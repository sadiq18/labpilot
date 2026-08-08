"""Shared SQLite connection client for ``knowledge.db``.

Owns the connection conventions that used to live inside the Knowledge Store:
``sqlite3.Row`` row factory, ``PRAGMA foreign_keys = ON``, and running the
unified migration. Domain stores (KnowledgeStore, PlanStore, …) stay
pillar-owned but take their connection from here so schema location and
connection setup never drift between pillars.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from labpilot.accessor.sqlite.migrate import run_migration


class SqliteClient:
    """Open (and migrate) one competition's ``knowledge.db``.

    ``conn`` is a live :class:`sqlite3.Connection` with ``Row`` factory and
    foreign keys enabled — domain stores execute their queries against it
    directly.

    ``allow_cross_thread`` is **opt-in per caller**, deliberately, and defaults
    to the thread-confined behaviour. Domain stores run their own SQL against
    `conn` without taking any lock, so flipping this on globally would make
    cross-thread use *possible* everywhere while making it *safe* nowhere —
    sqlite tolerates cross-thread use, not concurrent use. A caller that opts
    in owns the serialisation, as `BudgetLedger` and `PromptCache` already do.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        migrate: bool = True,
        allow_cross_thread: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=not allow_cross_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if migrate:
            self.migrate()

    def migrate(self) -> None:
        run_migration(self.conn)

    def schema_version(self) -> str:
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return row["value"] if row else ""

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SqliteClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
