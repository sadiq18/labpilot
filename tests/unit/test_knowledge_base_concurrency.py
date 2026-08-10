"""M11: KnowledgeBase is a whole-file read-modify-write store.

Each instance loads the entire entry dict at construction, mutates it in
memory, and writes the whole file back. Two branches doing that concurrently
means the second write discards *every* entry the first added — not a
conflict on one field, a total loss of the other branch's work.

Found by enumeration (`grep -rln "_load()\\|_save()"`), not by the three
adversarial review rounds that preceded it, and it was the only remaining
instance of this shape in the engine.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.research_engine.shared.experiments.knowledge import KnowledgeBase
from labpilot.research_engine.shared.experiments.models import (
    ExperimentComparison,
    KnowledgeEffect,
    Verdict,
)


def _comparison(compare_id: str, delta: float) -> ExperimentComparison:
    return ExperimentComparison(
        base_id="run-base",
        compare_id=compare_id,
        primary_metric_key="cv_accuracy",
        metric_deltas={"cv_accuracy": delta},
        changes=[],
        runtime_delta_seconds=None,
        runtime_delta_pct=None,
        verdict=Verdict.WORTH_KEEPING,
        verdict_reason="test",
    )


def test_concurrent_updates_do_not_lose_entries(tmp_path: Path) -> None:
    """Each branch gets its OWN KnowledgeBase, as parallel branches would."""
    n_branches = 8
    barrier = threading.Barrier(n_branches)

    def branch(i: int) -> None:
        # Constructed per branch — this is the snapshot that goes stale.
        kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
        barrier.wait()
        kb.update_from_comparison(
            _comparison(f"cmp-{i}", 0.01),
            technique_tags=[f"technique_{i}"],
        )

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(n_branches)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every branch's entry must survive. Without the locked reload, the last
    # writer's stale snapshot overwrites the file and most of these vanish.
    final = KnowledgeBase(tmp_path / "knowledge", "titanic")
    survived = {e.technique for e in final.list_entries()}
    expected = {f"technique_{i}" for i in range(n_branches)}
    assert survived == expected, f"lost {sorted(expected - survived)}"


def test_repeated_updates_accumulate_sample_size(tmp_path: Path) -> None:
    """Concurrent updates to the SAME technique must accumulate, not clobber.

    Each update reads the existing entry to increment sample_size, so this is
    the read-modify-write path specifically, not just distinct-key survival.
    """
    n_updates = 8
    barrier = threading.Barrier(n_updates)

    def branch(i: int) -> None:
        kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
        barrier.wait()
        kb.update_from_comparison(
            _comparison(f"cmp-{i}", 0.01),
            technique_tags=["shared_technique"],
        )

    threads = [threading.Thread(target=branch, args=(i,)) for i in range(n_updates)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = KnowledgeBase(tmp_path / "knowledge", "titanic")
    entry = final.get("shared_technique", "cv_accuracy")
    assert entry is not None
    assert entry.sample_size == n_updates
    assert len(entry.evidence_run_ids) == n_updates


def test_save_is_atomic_no_torn_reads(tmp_path: Path) -> None:
    """A reader must never observe a partial knowledge_base.json."""
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    kb.update_from_comparison(
        _comparison("cmp-seed", 0.01),
        technique_tags=["seed_technique"],
    )
    stop = threading.Event()
    torn_reads: list[str] = []

    def writer() -> None:
        for i in range(100):
            kb.update_from_comparison(
                _comparison(f"cmp-{i}", 0.01),
                technique_tags=[f"technique_{i}"],
            )
        stop.set()

    def reader() -> None:
        while not stop.is_set():
            try:
                text = kb.path.read_text()
            except FileNotFoundError:
                continue
            if not text.strip().endswith("}"):
                torn_reads.append(text)

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(4)]
    writer_thread.start()
    for t in reader_threads:
        t.start()
    writer_thread.join()
    for t in reader_threads:
        t.join()

    assert torn_reads == []


def test_effect_still_derived_after_locked_reload(tmp_path: Path) -> None:
    """Guard the refactor didn't change single-threaded semantics."""
    kb = KnowledgeBase(tmp_path / "knowledge", "titanic")
    updated = kb.update_from_comparison(
        _comparison("cmp-1", 0.05),
        technique_tags=["mixup"],
    )
    assert len(updated) == 1
    assert updated[0].technique == "mixup"
    assert updated[0].effect == KnowledgeEffect.IMPROVES

    reloaded = KnowledgeBase(tmp_path / "knowledge", "titanic")
    assert reloaded.get("mixup", "cv_accuracy") is not None
