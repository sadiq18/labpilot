"""A branch opens its own `ConductorStore`; K of them at once must all work.

The alternative — one campaign-scoped store shared by every branch — cannot
work as written: `SqliteClient` leaves `check_same_thread` on, and a branch's
work runs on a worker thread (`agents/experiment.py` offloads via
`anyio.to_thread.run_sync`), so the shared connection raises there. Opening
per branch is also what the rest of the codebase already does with
`KnowledgeStore`, and it keeps holding when a branch becomes a subprocess,
which the worktree design anticipates.

What that costs is concurrency at open: the schema migration runs on every
connect, and it is not idempotent when two openers race it.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from labpilot.accessor.sqlite.client import SqliteClient
from labpilot.research_engine.conductor.store import ConductorStore


def _run_concurrently(fn, count: int = 8) -> list[str]:
    """Run `fn(i)` on `count` threads released together; return the errors."""
    errors: list[str] = []
    guard = threading.Lock()
    start = threading.Barrier(count)

    def one(index: int) -> None:
        start.wait()
        try:
            fn(index)
        except Exception as exc:  # noqa: BLE001 — the failures are the result
            with guard:
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=one, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_eight_branches_open_their_own_store_at_once(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    ConductorStore(knowledge, "titanic").close()

    errors = _run_concurrently(lambda _: ConductorStore(knowledge, "titanic").close())

    assert errors == []


def test_the_store_closes_itself_when_used_as_a_context_manager(
    tmp_path: Path,
) -> None:
    """Per-branch open-use-close is only safe if closing is automatic; every
    `KnowledgeStore` call site already relies on this shape."""
    knowledge = tmp_path / "knowledge"
    with ConductorStore(knowledge, "titanic") as store:
        session = store.create_session("beat baseline")
        assert session.id

    try:
        store._conn.execute("SELECT 1")
    except sqlite3.ProgrammingError as exc:
        assert "closed" in str(exc).lower()
    else:  # pragma: no cover - only reached if __exit__ stopped closing
        raise AssertionError("the connection was left open")


def test_the_context_manager_closes_even_when_the_body_raises(
    tmp_path: Path,
) -> None:
    knowledge = tmp_path / "knowledge"
    store = ConductorStore(knowledge, "titanic")
    try:
        with store:
            raise RuntimeError("branch failed")
    except RuntimeError:
        pass

    try:
        store._conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a failing branch leaked its connection")


def test_branches_racing_an_unmigrated_database_do_not_collide(
    tmp_path: Path, monkeypatch
) -> None:
    """The migration path K branches hit when they are first to a stale DB.

    `run_migration` reads whether `techniques.status` exists and then adds it.
    Two openers that both read "absent" both run the `ALTER`, and the loser
    gets `duplicate column name`.

    Left to chance this reproduces about one run in five, so the window is
    held open deliberately: every thread parks between its check and its
    `ALTER` until all eight have checked. The `ALTER`s themselves still run
    for real. Serialised, the barrier can never fill — the first holder times
    out, breaks it, and the rest pass straight through, which is why the
    timeout is a broken barrier rather than a hang.
    """
    db_path = tmp_path / "stale.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE techniques (
            id   TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    from labpilot.accessor.sqlite import migrate as migrate_mod

    real_has_column = migrate_mod._table_has_column
    checked = threading.Barrier(8)

    def parked_has_column(conn, table: str, column: str) -> bool:
        answer = real_has_column(conn, table, column)
        if (table, column) == ("techniques", "status"):
            try:
                checked.wait(timeout=0.5)
            except threading.BrokenBarrierError:
                pass  # serialised, as intended
        return answer

    monkeypatch.setattr(migrate_mod, "_table_has_column", parked_has_column)

    errors = _run_concurrently(lambda _: SqliteClient(db_path).close())

    assert errors == []
    check = sqlite3.connect(db_path)
    columns = {row[1] for row in check.execute("PRAGMA table_info(techniques)")}
    check.close()
    assert "status" in columns


def test_branches_racing_the_wal_switch_all_open(tmp_path: Path) -> None:
    """The other half of a first open: turning WAL on.

    `PRAGMA journal_mode = WAL` needs an exclusive lock and returns
    SQLITE_BUSY *without* invoking the busy handler, so `busy_timeout` does
    not cover it. Eight branches reaching a not-yet-WAL database together is
    the shape a fan-out produces on a fresh competition.
    """
    db_path = tmp_path / "fresh.db"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("CREATE TABLE seed (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()

    errors = _run_concurrently(lambda _: SqliteClient(db_path).close())

    assert errors == []
    check = sqlite3.connect(db_path)
    mode = check.execute("PRAGMA journal_mode").fetchone()[0]
    check.close()
    assert str(mode).lower() == "wal"


def test_concurrent_writes_from_per_branch_stores_all_land(tmp_path: Path) -> None:
    """The point of the whole arrangement: eight branches, eight rows."""
    knowledge = tmp_path / "knowledge"
    with ConductorStore(knowledge, "titanic") as main:
        session = main.create_session("beat baseline")

    def enqueue(_index: int) -> None:
        with ConductorStore(knowledge, "titanic") as store:
            store.enqueue(session.id, "run_plan")

    errors = _run_concurrently(enqueue)

    assert errors == []
    with ConductorStore(knowledge, "titanic") as store:
        tasks = store.list_tasks(session_id=session.id)
    assert len({task.id for task in tasks}) == 8
