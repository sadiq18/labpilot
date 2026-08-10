"""M11: concurrent EvidenceCardStore.save() calls must not collide on id.

Same TOCTOU shape as `ConductorStore.new_decision_id`, over the filesystem
instead of SQLite: `new_id()` globs `EV-*.json` for the current max and
computes `max+1` with no lock, so two concurrent `save()` calls for new cards
can both compute the same next id before either file lands on disk.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.research_engine.evidence.models import EvidenceCard
from labpilot.research_engine.evidence.store import EvidenceCardStore


def test_concurrent_save_of_new_cards_does_not_collide(tmp_path: Path) -> None:
    store = EvidenceCardStore(tmp_path / "knowledge", "titanic")
    n_writers = 8
    saved_ids: list[str] = []
    lock = threading.Lock()

    def save_one() -> None:
        card = store.save(EvidenceCard(treatment_experiment="run-x"))
        with lock:
            saved_ids.append(card.id)

    threads = [threading.Thread(target=save_one) for _ in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(saved_ids) == n_writers
    assert len(set(saved_ids)) == n_writers
    on_disk = list((store.dir).glob("EV-*.json"))
    assert len(on_disk) == n_writers
