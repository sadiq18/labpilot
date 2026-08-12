"""Enqueueing K branch tasks at once must produce K tasks (M11 task 7).

A task is what a branch *is*, so K-way fan-out enqueues K of them
concurrently. Each branch holds its own `ConductorStore`, because the
sqlite connection is thread-confined (`SqliteClient.allow_cross_thread`
defaults off and `ConductorStore` does not opt in) — so the serialisation
has to come from the cross-process file lock, not from sharing a connection.

Measured before the fix: 8 concurrent enqueues produced one task and seven
`UNIQUE constraint failed: os_tasks.id`.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.research_engine.conductor.store import ConductorStore


def test_eight_branches_enqueueing_at_once_get_eight_distinct_tasks(
    tmp_path: Path,
) -> None:
    knowledge = tmp_path / "knowledge"
    main = ConductorStore(knowledge, "titanic")
    session = main.create_session("beat baseline")
    main.close()

    ids: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def enqueue_one() -> None:
        store = ConductorStore(knowledge, "titanic")
        start.wait()
        try:
            task = store.enqueue(session.id, "run_plan")
            with lock:
                ids.append(task.id)
        except Exception as exc:  # noqa: BLE001 — the failure under test
            with lock:
                errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            store.close()

    threads = [threading.Thread(target=enqueue_one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(ids) == 8
    assert len(set(ids)) == 8


def test_a_single_enqueue_is_unchanged(tmp_path: Path) -> None:
    """The sequential path keeps its ids and its shape."""
    store = ConductorStore(tmp_path / "knowledge", "titanic")
    try:
        session = store.create_session("g")

        first = store.enqueue(session.id, "run_plan", args={"plan_id": "P-001"})
        second = store.enqueue(session.id, "reflect", dependencies=[first.id])

        assert first.id == "T-001"
        assert second.id == "T-002"
        assert first.args == {"plan_id": "P-001"}
        assert second.dependencies == [first.id]
        assert store.get_task(first.id) is not None
    finally:
        store.close()
