"""Shared SQLite connection client for ``knowledge.db``.

Owns the connection conventions that used to live inside the Knowledge Store:
``sqlite3.Row`` row factory, ``PRAGMA foreign_keys = ON``, and running the
unified migration. Domain stores (KnowledgeStore, PlanStore, …) stay
pillar-owned but take their connection from here so schema location and
connection setup never drift between pillars.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from labpilot.accessor.sqlite.migrate import run_migration


@contextmanager
def write_lock_for(db_path: str | Path) -> Iterator[None]:
    """Cross-process lock for one database file's multi-statement writes (M11).

    A single atomic statement (e.g. ``SET field = field + ?``) is already
    safe under WAL + busy_timeout without any lock; this exists for
    multi-statement sequences — allocate-an-id-then-insert-a-row being the
    concrete case in `ConductorStore` — where two callers could otherwise
    both read the same "next id" before either writes.

    Built on `accessor.common.file_lock.locked` (`fcntl.flock`), the same
    primitive `HypothesisStore`/`EvidenceCardStore` use — not a
    `threading.RLock`. M11 branches are separate git worktrees and may end up
    as separate OS processes, not just threads; an in-process lock would
    silently stop protecting anything the moment that happens, while this
    keeps working either way. Not reentrant — do not nest two `with
    write_lock_for(...)` blocks for the same `db_path` on the same thread,
    unlike the `RLock` this replaced.

    Keyed by resolved `db_path` (one lock file alongside the database, not a
    single lock for the whole process): two unrelated competitions' stores,
    backed by physically distinct files, have zero real contention and
    should not serialize on each other just because both happened to call
    this at the same moment.
    """
    from labpilot.accessor.common.file_lock import locked

    resolved = Path(db_path).resolve()
    lock_path = resolved.with_name(resolved.name + ".writelock")
    with locked(lock_path):
        yield


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
    or takes `write_lock_for(self.db_path)` above for multi-statement writes.

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
        # `busy_timeout` before anything that can contend, so the busy handler
        # is armed for the rest of this constructor.
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA foreign_keys = ON")
        # One lock over the WAL switch *and* the migration (M11). Both are
        # writes that K branches opening their own store would otherwise run
        # at once, and neither is safe to race:
        #
        # * `PRAGMA journal_mode = WAL` needs an exclusive lock and returns
        #   SQLITE_BUSY without invoking the busy handler, so `busy_timeout`
        #   does not cover it — a branch could lose the open to `database is
        #   locked` before running a single query. (Skipping it when the mode
        #   is already WAL was measured *slower* than reissuing it: the read
        #   costs more than the no-op write.);
        # * `run_migration` checks whether a column exists and then adds it,
        #   so two openers that both read "absent" both run the `ALTER` and
        #   the loser gets `duplicate column name`.
        #
        # Serialising the whole block rather than guarding each statement
        # means the next migration added inherits the safety. Costs ~1ms per
        # open, against a step that runs a training job.
        with write_lock_for(self.db_path):
            self.conn.execute("PRAGMA journal_mode = WAL")
            if migrate:
                self._migrate_unlocked()

    def _migrate_unlocked(self) -> None:
        """`migrate()` without taking the lock — the caller already holds it.

        Separate because `write_lock_for` is not reentrant, so the constructor
        cannot reach the public method from inside its own lock.
        """
        run_migration(self.conn)

    def migrate(self) -> None:
        """Apply the schema, serialised against other openers (M11)."""
        with write_lock_for(self.db_path):
            self._migrate_unlocked()

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
