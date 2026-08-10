"""M11: concurrent writers against ConductorStore must not lose or corrupt rows.

Two different claims, verified differently:

- ``increment_metric``'s ``SET field = field + ?`` is a single atomic SQL
  statement — SQLite guarantees no lost updates for it regardless of an
  application-level lock. This test asserts that guarantee holds under real
  thread contention (each thread opens its own connection, matching M11's
  one-branch-per-thread model), not that it fixes a reproducible bug — a
  30-thread barrier-synchronized run against this same statement produced
  zero `database is locked` errors even before WAL/busy_timeout were added
  (`sqlite3.connect`'s implicit 5s timeout already covers that case here).
- Allocating an id then inserting a row using it (``new_decision_id`` +
  ``append_decision``) is a genuine multi-statement TOCTOU race no amount of
  busy_timeout closes — two callers can both read the same "next id" before
  either writes. This one needs the shared `write_lock` held across the whole
  sequence, and is the actual bug this file exists to guard against.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.accessor.sqlite.client import write_lock
from labpilot.research_engine.conductor.models import DecisionRecord
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace

_N_WRITERS = 8


def _ws(tmp_path: Path, slug: str = "concurrency-demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def test_concurrent_decision_ids_do_not_collide(tmp_path: Path) -> None:
    """M11: N branches each appending their own DecisionRecord get N distinct ids.

    Each thread opens its **own** `ConductorStore` (own connection) — exactly
    what M11's K-way fan-out does (one branch per thread via
    `anyio.to_thread.run_sync`, each branch building its own store, the same
    pattern `KnowledgeStore` already uses). A shared `sqlite3.Connection`
    cannot cross threads at all (`check_same_thread`), so the real hazard
    under test is N separate connections against the same file, not one
    connection used unsafely.
    """
    ws = _ws(tmp_path)
    setup_store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = setup_store.create_session("parallel branches")
    finally:
        setup_store.close()

    def append_one() -> None:
        store = ConductorStore(ws.knowledge_dir, ws.competition)
        try:
            with write_lock:
                decision_id = store.new_decision_id()
                store.append_decision(
                    DecisionRecord(
                        id=decision_id,
                        session_id=session.id,
                        tool_name="run_experiment",
                        rationale="branch",
                    )
                )
        finally:
            store.close()

    threads = [threading.Thread(target=append_one) for _ in range(_N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    verify_store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        decisions = verify_store.list_decisions(session.id)
        assert len(decisions) == _N_WRITERS
        assert len({d.id for d in decisions}) == _N_WRITERS
    finally:
        verify_store.close()


def test_concurrent_increment_metric_loses_nothing(tmp_path: Path) -> None:
    """M11: a single atomic UPDATE is safe under WAL+busy_timeout without a lock.

    Each thread opens its own `ConductorStore`, same rationale as above.
    """
    ws = _ws(tmp_path)
    setup_store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = setup_store.create_session("parallel branches")
    finally:
        setup_store.close()

    def increment() -> None:
        store = ConductorStore(ws.knowledge_dir, ws.competition)
        try:
            store.increment_metric(session.id, "tasks_failed", 1)
        finally:
            store.close()

    threads = [threading.Thread(target=increment) for _ in range(_N_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    verify_store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        metrics = verify_store.get_metrics(session.id)
        assert metrics.tasks_failed == _N_WRITERS
    finally:
        verify_store.close()
