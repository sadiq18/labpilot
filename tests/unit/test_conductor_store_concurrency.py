"""M11: concurrent writers against ConductorStore must not lose or corrupt rows.

Two different claims, verified differently:

- Allocating an id then inserting a row using it (``new_decision_id`` +
  ``append_decision``) is a genuine multi-statement TOCTOU race no amount of
  busy_timeout closes — two callers can both read the same "next id" before
  either writes. `ConductorStore.append_new_decision` closes it by holding
  `write_lock_for(db_path)` across the whole sequence; this test drives that
  method directly (not a hand-rolled lock in the test) so it proves what
  production code — M11 task 7's K-way fan-out — will actually call.
- ``increment_metric`` looked like a single atomic SQL statement but wasn't,
  on a session's first increment: it used to do
  ``get_metrics()`` (SELECT) → conditionally ``upsert_metrics()`` (its own
  commit) → the ``UPDATE``, three separately-committed statements. Two
  threads' first increments could both see no row, then race the upsert's
  ``ON CONFLICT DO UPDATE``, and whichever committed second reset every
  field — including a sibling's already-committed increment — back to zero.
  Fixed with a single atomic ``INSERT OR IGNORE`` ahead of the ``UPDATE``.
  This test's setup deliberately does not pre-create the metrics row, so it
  exercises exactly the path that used to be unsafe.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.accessor.sqlite.client import write_lock_for
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
            store.append_new_decision(session.id, "run_experiment", "branch")
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
    """M11: increment_metric is safe under concurrency, including a session's
    first increment (the INSERT OR IGNORE ahead of the UPDATE), with no lock.

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


def test_write_lock_for_is_scoped_per_db_path(tmp_path: Path) -> None:
    """M11: unrelated competitions must not serialize on each other's writes."""
    ws_a = _ws(tmp_path, "competition-a")
    ws_b = _ws(tmp_path, "competition-b")
    store_a = ConductorStore(ws_a.knowledge_dir, ws_a.competition)
    store_b = ConductorStore(ws_b.knowledge_dir, ws_b.competition)
    try:
        assert write_lock_for(store_a.paths.db_path) is write_lock_for(store_a.paths.db_path)
        assert write_lock_for(store_a.paths.db_path) is not write_lock_for(store_b.paths.db_path)
    finally:
        store_a.close()
        store_b.close()
