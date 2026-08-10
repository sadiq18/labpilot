"""Shared SQLite connection client for ``knowledge.db``.

Owns the connection conventions that used to live inside the Knowledge Store:
``sqlite3.Row`` row factory, ``PRAGMA foreign_keys = ON``, and running the
unified migration. Domain stores (KnowledgeStore, PlanStore, …) stay
pillar-owned but take their connection from here so schema location and
connection setup never drift between pillars.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from labpilot.accessor.sqlite.migrate import run_migration

#: Shared across every :class:`SqliteClient` instantiation in the process
#: (M11) — an instance-level lock does not serialize anything here, since
#: `ConductorStore`/`KnowledgeStore` construct a fresh client at each call
#: site rather than sharing one long-lived object the way `BudgetLedger`
#: does. A single atomic statement (e.g. ``SET field = field + ?``) is
#: already safe under WAL + busy_timeout without this lock; it exists for
#: multi-statement sequences — allocate-an-id-then-insert-a-row being the
#: concrete case in `ConductorStore` — where two callers could otherwise
#: both read the same "next id" before either writes.
write_lock = threading.RLock()


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
    in owns the serialisation, as `BudgetLedger` and `PromptCache` already do,
    or takes the module-level `write_lock` above for multi-statement writes.

    WAL journal mode plus an explicit ``busy_timeout`` (M11) is for read/write
    concurrency during a parallel step, not a fix for a reproducible
    ``database is locked`` failure — ``sqlite3.connect`` already carries an
    implicit 5s retry via its own ``timeout`` parameter, so that exception was
    not actually being hit by this codebase's default rollback-journal setup.
    WAL removes the "readers block behind an in-flight writer" behavior that
    mode has, and setting `busy_timeout` explicitly makes the retry window a
    stated contract instead of an implicit driver default.
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
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
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
