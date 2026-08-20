"""Nine states, derived on read, and nothing withheld yet.

M23 step 5. The floor and Baseline 1 are measurements; this is the verdict over
them — and it is **observe-only**. Enforcement is step 8, deliberately last: the
plan's own trap records that `_observe_delta` was *"calibrated against
hand-written samples, and that is precisely how the two bugs got in"*, so one
campaign's worth of recorded verdicts is what turns "the gate is right" from an
argument into a false-positive rate.

**Nine states rather than a boolean**, because each has a different operator
action and M20's finding is that collapsing them is how eight gates reported
`pass` on things that could not run:

| state | what the operator does |
|---|---|
| `unknown` | nothing has been measured; run the baseline |
| `floor_missing` | no reading on disk; run the baseline |
| `floor_undefined` | this target has no defined floor; nothing to do here |
| `blocked_uncertain` | **answer the schema question** — do not debug a baseline |
| `awaiting_ml` | Baseline 1 cannot run here; the floor still stands |
| `stale` | the dataset or the answers moved; re-measure |
| `failed` | the pipeline loses to a constant; read the report |
| `passed` | proceed |
| `waived` | someone accepted `failed` in writing, and it is recorded |

**Derived on read, never stored.** A stored verdict is derived state that
outlives its cause — AGENTS.md rule 2, and the mistake `apply_card_to_beliefs`
cost this repo. The readings are facts; the verdict is a function of them.

**It does not read `H-BASELINE.status`.** Five layers of derivation, one of which
raises, so a bookkeeping fault would read as "baseline not passed". It reads the
floor reading, written by one writer. H-BASELINE finally getting a status is a
valuable consequence, cross-checked in tests, never a dependency.

Plan: ``docs/research-os/autonomy-roadmap/design/18-baseline-correctness.md`` §7.3
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from labpilot.research_engine.execution.baseline.baseline_one import (
    BaselineComparison,
    ModelReading,
    compare,
    load_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import FloorReading, load_floor

logger = logging.getLogger(__name__)

__all__ = [
    "GATE_STATES",
    "WAIVER_FILENAME",
    "GateVerdict",
    "Waiver",
    "evaluate_gate",
    "load_waiver",
    "reading_fingerprint",
    "write_waiver",
]

GateState = Literal[
    "unknown",
    "floor_missing",
    "floor_undefined",
    "blocked_uncertain",
    "awaiting_ml",
    "stale",
    "failed",
    "passed",
    "waived",
]

#: Ordered as the checks run, which is also roughly "least measured" to "most".
GATE_STATES: tuple[GateState, ...] = (
    "unknown",
    "floor_missing",
    "blocked_uncertain",
    "stale",
    "floor_undefined",
    "awaiting_ml",
    "failed",
    "passed",
    "waived",
)


#: States in which a campaign has *not* demonstrated a working baseline. Written
#: as the complement of `passed`/`waived` rather than as a list, so a tenth state
#: added later defaults to blocking rather than to permitted — the direction that
#: fails safe.
def blocks_research(state: GateState) -> bool:
    """Whether this state should stop hypothesis minting, once step 8 enforces."""
    return state not in ("passed", "waived")


WAIVER_FILENAME = "baseline_waiver.json"


class Waiver(BaseModel):
    """A recorded decision to proceed over a `failed` gate.

    Durable and specific: it names the fingerprint it was granted against, so a
    re-profiled dataset or a changed answer invalidates it. A waiver that
    outlived its cause would be the gate quietly switching itself off, which is
    the failure the env-var kill switch was rejected for.
    """

    reason: str
    granted_by: str = ""
    granted_at: str = ""
    #: What was true when it was granted. A waiver whose fingerprint no longer
    #: matches is not a waiver for *this* dataset.
    fingerprint: str = ""


class GateVerdict(BaseModel):
    """The state, why it holds, and what the operator does about it."""

    state: GateState = "unknown"
    reason: str = ""
    comparison: BaselineComparison = Field(default_factory=BaselineComparison)
    fingerprint: str = ""
    #: False while the rollout is observe-only. The verdict is recorded and
    #: reported; nothing is withheld. Step 8 flips this.
    enforced: bool = False
    evaluated_at: str = ""

    @property
    def blocks_research(self) -> bool:
        """Whether hypothesis minting *would* stop — subject to `enforced`."""
        return blocks_research(self.state)

    @property
    def withholds_anything(self) -> bool:
        """Whether anything is actually withheld right now.

        Two properties rather than one: "this campaign has not shown a working
        baseline" and "the system is refusing it something" are different facts,
        and conflating them is how an observe-only rollout would quietly become
        an enforcing one.
        """
        return self.enforced and self.blocks_research


def reading_fingerprint(root: Path) -> str:
    """What the readings must describe to still be current.

    Covers the validation scheme, the target, the metric, `profile.schema_version`
    **and the M22 answers fingerprint**. The last one is the point: an operator
    answering *"the label is `Depth`"* invalidates every reading that described
    `Zone_Depth`, and without it the gate would keep reporting `passed` over a
    measurement of the wrong column.
    """
    root = Path(root)

    def _read(name: str) -> dict:
        path = root / name
        if not path.is_file():
            return {}
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return body if isinstance(body, dict) else {}

    profile, choice = _read("profile.json"), _read("baseline_choice.json")
    plan = choice.get("validation") if isinstance(choice.get("validation"), dict) else {}
    digest = hashlib.sha256()
    for part in (
        str(plan.get("scheme", "")),
        str(plan.get("group_key", "")),
        str(plan.get("n_splits", "")),
        str(choice.get("target_column") or profile.get("target_column") or ""),
        str(choice.get("metric_name", "")),
        str(profile.get("schema_version", "")),
        str(profile.get("answers_fingerprint", "")),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _open_questions(root: Path) -> list[str]:
    """Blocking schema fields nobody has settled, from the shipped path."""
    try:
        from labpilot.accessor.profiler.questions import load_answers, pending_schema_questions
        from labpilot.accessor.profiler.tabular import DatasetProfile

        path = Path(root) / "profile.json"
        if not path.is_file():
            return []
        profile = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
        return [q.field for q in pending_schema_questions(profile, load_answers(Path(root)))]
    except Exception as exc:  # noqa: BLE001 — an unreadable profile is not a question
        logger.info("Could not read schema questions: %s", exc)
        return []


def evaluate_gate(
    root: Path,
    *,
    direction: str = "",
    enforced: bool = False,
    floor: FloorReading | None = None,
    model: ModelReading | None = None,
) -> GateVerdict:
    """The verdict for this workspace, computed from what is on disk.

    Order matters and is the design's: an uncertain schema outranks everything,
    because a floor computed against a guessed target is worse than no floor —
    that is goal 8, and `blocked_uncertain` was in the plan's nine states with no
    defined trigger until M22 gave it one.
    """
    root = Path(root)
    now = datetime.now(UTC).isoformat()
    fingerprint = reading_fingerprint(root)
    verdict = GateVerdict(fingerprint=fingerprint, enforced=enforced, evaluated_at=now)

    if floor is None:
        floor = load_floor(root)
    if model is None:
        model = load_baseline_one(root)

    # 1. The schema first. An open question means the target may be wrong, and
    #    every number below it would be a measurement of the wrong column. The
    #    operator answers; they do not debug a baseline.
    questions = _open_questions(root)
    if questions:
        verdict.state = "blocked_uncertain"
        verdict.reason = (
            f"the schema has not settled {', '.join(questions)}; a floor computed "
            "against a guessed target measures nothing"
        )
        return verdict

    if floor is None:
        verdict.state = "floor_missing"
        verdict.reason = f"no {'baseline_floor.json'} in this workspace; run the baseline"
        return verdict

    # 2. Did the dataset move under the reading? Asked before what it says,
    #    because a stale `passed` is more dangerous than no verdict at all.
    stale_reason = _staleness(floor, model, fingerprint)
    if stale_reason:
        verdict.state = "stale"
        verdict.reason = stale_reason
        return verdict

    if not floor.is_defined:
        verdict.state = "floor_undefined"
        verdict.reason = floor.undefined_reason or "no floor could be computed for this target"
        return verdict

    if model is None or not model.is_defined:
        verdict.state = "awaiting_ml"
        verdict.reason = (
            model.undefined_reason if model else ""
        ) or "no Baseline 1 reading; the floor stands but nothing has been compared to it"
        return verdict

    resolved = direction or _direction_for(floor.metric_name)
    verdict.comparison = compare(floor, model, resolved)
    if verdict.comparison.incomparable_reason:
        verdict.state = "floor_undefined"
        verdict.reason = verdict.comparison.incomparable_reason
        return verdict

    if verdict.comparison.beats_floor:
        verdict.state = "passed"
        verdict.reason = "the model beats the floor under the plan they share"
        return verdict

    verdict.state = "failed"
    verdict.reason = "the model does not beat a constant under the plan they share"

    waiver = load_waiver(root)
    if waiver is not None and waiver.fingerprint == fingerprint:
        verdict.state = "waived"
        verdict.reason = f"failed, waived: {waiver.reason}"
    elif waiver is not None:
        # Recorded, not honoured. A waiver granted against different inputs is
        # not a waiver for this dataset, and silently ignoring it would leave an
        # operator believing they had already dealt with this.
        verdict.reason += "; a waiver exists but was granted against a different fingerprint"
    return verdict


def _staleness(floor: FloorReading, model: ModelReading | None, fingerprint: str) -> str:
    """Why the readings no longer describe this workspace, or empty.

    Read from the readings themselves rather than a stamp file beside them: a
    separate file is a second copy of the same fact, free to disagree with the
    reading it describes, and nothing would keep the two in step.

    An unstamped reading is *not* treated as stale. Every reading written before
    this field existed has an empty one, and re-measuring every workspace on
    upgrade would train an operator to ignore the state — the design's own point
    about `stale` mattering as much as `failed`.
    """
    for name, recorded in (
        ("floor", floor.workspace_fingerprint),
        ("model", model.workspace_fingerprint if model else ""),
    ):
        if recorded and fingerprint and recorded != fingerprint:
            return (
                f"the {name} reading describes a different workspace: the target, "
                "metric, plan, profile version or answers have changed since it was taken"
            )
    return ""


def _direction_for(metric_name: str) -> str:
    from labpilot.research_engine.intelligence.competition.metric_vocabulary import direction_of

    return direction_of(metric_name) or ""


def write_waiver(root: Path, waiver: Waiver) -> Path:
    path = Path(root) / WAIVER_FILENAME
    path.write_text(waiver.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_waiver(root: Path) -> Waiver | None:
    path = Path(root) / WAIVER_FILENAME
    if not path.is_file():
        return None
    try:
        return Waiver.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
