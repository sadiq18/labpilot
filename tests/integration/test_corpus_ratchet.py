"""The corpus is a gate, not a report.

M24 exit criteria 1 and 2. A rate with no corpus behind it cannot be reproduced,
and a rate with no floor under it cannot fail. `RATCHET.json` carries both: the
digest of the corpus the numbers were measured over, and the value each
criterion may not fall below.

The floor is **today's measured value**, deliberately. The plan: *"Asserting 95%
on day one makes the suite red and teaches everyone to ignore it."* 0.95 sits
beside the floor as the goal, so the gap is visible rather than aspired to.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from labpilot.accessor.benchmark.ledger import (
    GOAL,
    RATCHET_FILENAME,
    corpus_hash,
    load_ledger,
    rates_from,
    regressions,
    unrecorded_gains,
)
from labpilot.accessor.benchmark.score import CRITERIA, profile_and_score

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "competitions"


def _slugs() -> list[str]:
    return sorted(p.name for p in CORPUS.iterdir() if (p / "fixture.json").is_file())


def _measured() -> dict[str, float]:
    work = Path(tempfile.mkdtemp())
    cards = {}
    for slug in _slugs():
        card = profile_and_score(CORPUS / slug, work / slug)
        cards[slug] = {c: (card.verdict_for(c) or "—") for c in CRITERIA}
    return rates_from(cards)


def test_the_corpus_has_a_ledger() -> None:
    """Without one the score is a report. `RATCHET.json` is what makes it a gate."""
    ledger = load_ledger(CORPUS)

    assert ledger is not None, f"no {RATCHET_FILENAME} in the corpus"
    assert ledger.floors, "a ledger with no floors cannot fail"
    assert ledger.goal == GOAL


def test_no_criterion_has_fallen_below_its_floor() -> None:
    """The regression half of the ratchet, and the reason it exists."""
    ledger = load_ledger(CORPUS)
    assert ledger is not None

    fallen = regressions(ledger, _measured())

    assert not fallen, "the corpus has regressed:\n" + "\n".join(
        f"  {criterion}: floor {floor:.2f}, now {now:.2f}"
        for criterion, (floor, now) in sorted(fallen.items())
    )


def test_an_improvement_is_recorded_rather_than_absorbed() -> None:
    """The other half. Silently absorbing a gain is how a ratchet rots: the floor
    stays where it was, and the next regression falls into the slack the
    improvement left behind without tripping anything.

    This failing is good news with a chore attached — raise the floor in the
    commit that earned it.
    """
    ledger = load_ledger(CORPUS)
    assert ledger is not None

    gained = unrecorded_gains(ledger, _measured())

    assert not gained, "the corpus improved; raise the floors:\n" + "\n".join(
        f"  {criterion}: floor {floor:.2f} -> {now:.2f}"
        for criterion, (floor, now) in sorted(gained.items())
    )


def test_the_ledger_names_the_corpus_it_measured() -> None:
    """A rate with no corpus behind it is a number nobody can reproduce.

    Failing here means the fixtures changed — a capture, an expectation, a new
    competition — and the recorded rates describe a corpus that no longer
    exists. Re-measure and re-record together.
    """
    ledger = load_ledger(CORPUS)
    assert ledger is not None

    assert ledger.corpus_hash == corpus_hash(CORPUS), (
        "the corpus has changed since the floors were recorded; re-run the ratchet and commit both"
    )


def test_the_hash_is_stable_and_specific(tmp_path: Path) -> None:
    """Two runs agree; a changed manifest does not.

    Over `fixture.json` rather than the captured bytes, because the manifest
    already carries each file's `source_sha256` — so a re-capture moves it, and
    so does an edited expectation, which is the other half of what a score
    depends on.
    """
    import json
    import shutil

    assert corpus_hash(CORPUS) == corpus_hash(CORPUS)

    copy = tmp_path / "corpus"
    shutil.copytree(CORPUS, copy)
    assert corpus_hash(copy) == corpus_hash(CORPUS), "a copy is the same corpus"

    slug = _slugs()[0]
    manifest = copy / slug / "fixture.json"
    body = json.loads(manifest.read_text(encoding="utf-8"))
    body["expected"]["target_column"] = "something-else"
    manifest.write_text(json.dumps(body), encoding="utf-8")

    assert corpus_hash(copy) != corpus_hash(CORPUS), "an edited expectation is a new corpus"


@pytest.mark.parametrize("goal_stage", ["target_column", "train_test_relationship", "metric_name"])
def test_the_gap_to_the_goal_is_visible(goal_stage: str) -> None:
    """The plan's bar is >95% on the schema stages. Recorded beside the floor
    rather than asserted, so a stage below it is a fact in the ledger instead of
    a red suite nobody reads."""
    ledger = load_ledger(CORPUS)
    assert ledger is not None

    floor = ledger.floors.get(goal_stage)

    assert floor is not None, f"{goal_stage} is not in the ledger"
    if floor < ledger.goal:
        pytest.skip(f"{goal_stage} is at {floor:.2f}, below the {ledger.goal:.2f} goal — recorded")
