"""`create_unless_covered` must decide and write under one lock (M16).

Every minting path already had a duplicate check — `_already_covered_by_proposed`
in `execution/outcome.py` and three inline scans beside it — and every one of
them listed the proposed pool, decided, and *then* called `create()`, with
nothing held across the pair. With one writer that window never opened. M16 adds
a second (the background evidence producer), and the second reads the pool a
millisecond before the first writes to it.
"""

from __future__ import annotations

import threading
from pathlib import Path

from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

_TAG = "dedupe-me"


def _store(tmp_path: Path) -> HypothesisStore:
    return HypothesisStore(tmp_path / "knowledge", "titanic")


def _covers(tag: str):
    """"Already proposed" = an open proposal carries this tag."""

    def predicate(proposed) -> bool:
        return any(tag in h.tags for h in proposed)

    return predicate


def _mint(store: HypothesisStore, tag: str = _TAG, **overrides):
    fields = {
        "observation": "same idea",
        "reason": "same reason",
        "prediction": "same prediction",
        "confidence": 0.5,
        "tags": [tag],
    }
    fields.update(overrides)
    return store.create_unless_covered(covered_by=_covers(tag), **fields)


def test_the_second_mint_of_one_idea_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = _mint(store)
    second = _mint(store)

    assert first is not None
    assert second is None
    assert len(store.list(status=HypothesisStatus.PROPOSED)) == 1


def test_an_uncovered_idea_is_still_created(tmp_path: Path) -> None:
    """The predicate must be able to say no, or this is a write-blocker."""
    store = _store(tmp_path)

    assert _mint(store) is not None
    other = _mint(store, tag="something-else", observation="different")

    assert other is not None
    assert len(store.list(status=HypothesisStatus.PROPOSED)) == 2


def test_eight_racing_writers_produce_one_row(tmp_path: Path) -> None:
    """The M16 case: producer and consumer minting the same idea at once."""
    store = _store(tmp_path)
    results: list[object] = []
    guard = threading.Lock()
    start = threading.Barrier(8)

    def mint() -> None:
        start.wait()
        outcome = _mint(store)
        with guard:
            results.append(outcome)

    threads = [threading.Thread(target=mint) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert len([r for r in results if r is not None]) == 1
    assert len(store.list(status=HypothesisStatus.PROPOSED)) == 1


def test_the_check_then_create_shape_this_replaces_admits_a_duplicate(tmp_path: Path) -> None:
    """The defect being removed, run for real rather than argued about.

    This is the *old* shape — list, decide, then create, with the two threads
    interleaving in between — spelled out inline. If it ever stops producing
    two rows, the fixture has lost the ability to express the race and the test
    above stops proving anything.
    """
    store = _store(tmp_path)
    decided = threading.Barrier(2)

    def mint_the_old_way() -> None:
        covered = _covers(_TAG)(store.list(status=HypothesisStatus.PROPOSED))
        decided.wait()  # both threads have decided before either writes
        if covered:
            return
        store.create(
            observation="same idea",
            reason="same reason",
            prediction="same prediction",
            confidence=0.5,
            tags=[_TAG],
        )

    threads = [threading.Thread(target=mint_the_old_way) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.list(status=HypothesisStatus.PROPOSED)) == 2


def test_fields_are_normalised_exactly_as_create_normalises_them(tmp_path: Path) -> None:
    """It forwards to the same `_prepare`, so combo/stack derivation is shared."""
    store = _store(tmp_path)

    minted = store.create_unless_covered(
        covered_by=lambda _proposed: False,
        observation="o",
        reason="r",
        prediction="p",
        confidence=0.5,
        combo_techniques=["swa", "tta"],
    )

    assert minted is not None
    assert minted.technique == "swa+tta"
    assert minted.technique_stack == ["swa", "tta"]


# --- the producer's own write path -------------------------------------------


def _card(rank: int, *, technique: str = "", prediction: str = "p", title: str = "t"):
    from labpilot.research_engine.intelligence.hypothesis.models import (
        HypothesisRecommendation,
    )

    return HypothesisRecommendation(
        rank=rank,
        hypothesis_id="",
        title=title,
        observation="o",
        reason="r",
        prediction=prediction,
        technique=technique,
    )


def test_two_writers_persisting_the_same_card_produce_one_row(tmp_path: Path) -> None:
    """`persist_recommendations` had no store-side dedupe at all before M16."""
    from labpilot.research_engine.intelligence.hypothesis.persist import (
        persist_recommendations,
    )

    knowledge = tmp_path / "knowledge"
    start = threading.Barrier(2)
    kept: list[list[object]] = []
    guard = threading.Lock()

    def persist() -> None:
        start.wait()
        out = persist_recommendations(
            [_card(1, technique="SWA", prediction="swa will help")],
            knowledge_dir=knowledge,
            competition="titanic",
        )
        with guard:
            kept.append(out)

    threads = [threading.Thread(target=persist) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store = HypothesisStore(knowledge, "titanic")
    assert len(store.list(status=HypothesisStatus.PROPOSED)) == 1
    # The loser reports nothing new, so `new_count = len(...)` stays honest.
    assert sorted(len(k) for k in kept) == [0, 1]


def test_one_batch_may_carry_several_cards_for_the_same_technique(tmp_path: Path) -> None:
    """A regression the golden analyze snapshot caught: five `SpecAugment`
    cards from five different sources are five proposals, not one.
    """
    from labpilot.research_engine.intelligence.hypothesis.persist import (
        persist_recommendations,
    )

    knowledge = tmp_path / "knowledge"
    cards = [
        _card(i, technique="SpecAugment", prediction=f"framing {i}", title=f"t{i}")
        for i in range(1, 6)
    ]

    kept = persist_recommendations(cards, knowledge_dir=knowledge, competition="titanic")

    assert len(kept) == 5
    assert len(HypothesisStore(knowledge, "titanic").list(status=HypothesisStatus.PROPOSED)) == 5


def test_a_combination_card_is_deduped_like_every_other_kind(tmp_path: Path) -> None:
    """The identity has to be the *stored* technique, not the one passed in.

    A combination proposal arrives with `technique=""` and its members in
    `combo_techniques`; the store fills the first from the second. Comparing
    the raw fields put `("", "alpha+beta", …)` beside `("alphabeta",
    "alpha+beta", …)` and called them different ideas, so every combination
    card duplicated freely while every other kind was caught.
    """
    from labpilot.research_engine.intelligence.hypothesis.persist import (
        persist_recommendations,
    )

    knowledge = tmp_path / "knowledge"
    combo = _card(1, prediction="combine alpha and beta")
    combo = combo.model_copy(update={"technique": "", "combo_techniques": ["alpha", "beta"]})

    first = persist_recommendations([combo], knowledge_dir=knowledge, competition="titanic")
    second = persist_recommendations([combo], knowledge_dir=knowledge, competition="titanic")

    assert len(first) == 1
    assert second == []
    assert len(HypothesisStore(knowledge, "titanic").list(status=HypothesisStatus.PROPOSED)) == 1


def test_the_stored_technique_is_derived_by_one_rule(tmp_path: Path) -> None:
    """`derive_technique` is that rule; both the store and the dedupe use it."""
    from labpilot.research_engine.shared.experiments.hypothesis import derive_technique

    assert derive_technique("", ["alpha", "beta"]) == "alpha+beta"
    assert derive_technique("SWA", ["alpha"]) == "SWA"
    assert derive_technique("  ", []) is None

    store = _store(tmp_path)
    minted = store.create(
        observation="o", reason="r", prediction="p", confidence=0.5,
        combo_techniques=["alpha", "beta"],
    )
    assert minted.technique == derive_technique("", ["alpha", "beta"])
