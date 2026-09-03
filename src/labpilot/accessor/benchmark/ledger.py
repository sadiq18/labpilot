"""What the corpus scored, which corpus scored it, and the floor it may not fall below.

M24 exit criteria 1 and 2. A rate with no corpus behind it is a number nobody
can reproduce, and a rate with no floor under it is a report rather than a gate.

**The hash pins the evidence.** Two runs of the same corpus produce the same
digest; adding a fixture, re-capturing one, or editing an expectation changes it.
So "target_column 5/5" stops being a claim about the system and becomes a claim
about the system *over a named corpus* — which is the only version of it anyone
can check later.

**The ratchet starts where the corpus is, not where the plan wants it.** The
design is explicit: *"Asserting 95% on day one makes the suite red and teaches
everyone to ignore it."* So the floor is today's measured value, 0.95 is recorded
as the goal beside it, and the gap between them is visible rather than aspired to.

**It fails in both directions.** Falling below the floor is a regression. Rising
above it without updating the ledger is how a ratchet rots — the design says so
about `known_failure` and the same argument holds for a rate: an improvement
nobody recorded is an improvement the next regression can hide behind.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = [
    "GOAL",
    "RATCHET_FILENAME",
    "Ledger",
    "corpus_hash",
    "load_ledger",
    "rates_from",
    "regressions",
    "save_ledger",
    "unrecorded_gains",
]

RATCHET_FILENAME = "RATCHET.json"

#: The plan's target for the four schema stages. Recorded, never asserted — the
#: floor is what the suite enforces, and this is what the floor is climbing to.
GOAL = 0.95

#: Verdicts that count toward a rate. `unverifiable` and `not_applicable` are a
#: fixture declining to answer, and folding them into a denominator would let a
#: corpus improve its score by capturing less.
_SCOREABLE = ("pass", "fail", "known_failure")


class Ledger(BaseModel):
    """Per-criterion floors, and the corpus they were measured over."""

    corpus_hash: str = ""
    #: criterion → the pass rate at the time it was recorded.
    floors: dict[str, float] = Field(default_factory=dict)
    goal: float = GOAL
    recorded_at: str = ""
    note: str = ""


def corpus_hash(corpus: Path) -> str:
    """A digest of every fixture manifest in `corpus`.

    Over `fixture.json` rather than the captured bytes: the manifest already
    carries each file's `source_sha256`, so a re-capture that changed the data
    changes this, and so does an edited expectation — which is the other half of
    what a score depends on. Sorted by slug, so it does not depend on the order
    a filesystem happened to return.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(corpus).glob("*/fixture.json")):
        digest.update(path.parent.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rates_from(cards: dict[str, dict[str, str]]) -> dict[str, float]:
    """criterion → pass rate, over the fixtures that can score it.

    `cards` is slug → {criterion: verdict}, which is what both the CLI and the
    tests already hold — passing verdicts rather than `Scorecard` objects keeps
    this module free of the scorer it measures.
    """
    rates: dict[str, float] = {}
    per_criterion: dict[str, list[str]] = {}
    for verdicts in cards.values():
        for criterion, verdict in verdicts.items():
            per_criterion.setdefault(criterion, []).append(verdict)
    for criterion, verdicts in per_criterion.items():
        scored = [v for v in verdicts if v in _SCOREABLE]
        if scored:
            rates[criterion] = sum(1 for v in scored if v == "pass") / len(scored)
    return rates


def regressions(ledger: Ledger, rates: dict[str, float]) -> dict[str, tuple[float, float]]:
    """criterion → (floor, measured) where the corpus has fallen below its floor.

    A criterion the ledger has no floor for is not a regression: a newly added
    criterion has nowhere to fall from, and treating its absence as zero would
    make every addition look like a win.
    """
    return {
        criterion: (floor, rates[criterion])
        for criterion, floor in ledger.floors.items()
        if criterion in rates and rates[criterion] < floor
    }


def unrecorded_gains(ledger: Ledger, rates: dict[str, float]) -> dict[str, tuple[float, float]]:
    """criterion → (floor, measured) where the corpus now does better than recorded.

    Reported, because silently absorbing an improvement is how a ratchet rots:
    the floor stays where it was, and the next regression falls into the slack
    the improvement left behind without tripping anything.
    """
    return {
        criterion: (floor, rates[criterion])
        for criterion, floor in ledger.floors.items()
        if criterion in rates and rates[criterion] > floor
    }


def load_ledger(corpus: Path) -> Ledger | None:
    path = Path(corpus) / RATCHET_FILENAME
    if not path.is_file():
        return None
    try:
        return Ledger.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_ledger(corpus: Path, ledger: Ledger) -> Path:
    path = Path(corpus) / RATCHET_FILENAME
    path.write_text(json.dumps(ledger.model_dump(), indent=2, sort_keys=True) + "\n", "utf-8")
    return path
