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
    """

    def __init__(self, db_path: str | Path, *, migrate: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
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
