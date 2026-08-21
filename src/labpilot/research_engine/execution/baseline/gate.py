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
from typing import Literal, get_args

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
    "UNCHANGEABLE_STATES",
    "baseline_is_settled",
    "blocks_research",
    "enforcement_enabled",
    "refuses_minting",
    "WAIVER_FILENAME",
    "GateVerdict",
    "Waiver",
    "evaluate_gate",
    "refuse_hypothesis_minting",
    "load_waiver",
    "reading_fingerprint",
    "write_waiver",
]

#: Ordered as the checks run, which is also roughly "least measured" to "most".
GateState = Literal[
    "unknown",
    "blocked_uncertain",
    "floor_missing",
    "stale",
    "floor_undefined",
    "awaiting_ml",
    "failed",
    "passed",
    "waived",
]

#: Derived from the type rather than retyped beside it. Two lists of the same
#: nine strings agreed on the day they were written and nothing kept them in
#: step — a tenth state added to one would have silently narrowed the
#: parametrized coverage that iterates the other, so the tests would keep
#: passing while checking less.
GATE_STATES: tuple[GateState, ...] = get_args(GateState)


#: States that are facts about the *dataset*, not about the campaign's work. No
#: amount of running plans changes them: Baseline 1 cannot be made to run where
#: features are not columns, and a metric nobody catalogued does not acquire a
#: floor because you tried again.
#:
#: They are the reason a second predicate exists. `blocks_research` says the gate
#: is not open; these say there is nothing the campaign can do about that — and
#: treating the two as one pinned an image competition to recompiling its
#: baseline forever, which is the trap the design names by name.
UNCHANGEABLE_STATES: frozenset[str] = frozenset({"floor_undefined", "awaiting_ml"})


def blocks_research(state: GateState) -> bool:
    """Whether this state means the gate is not open.

    The complement of `passed`/`waived` rather than a list of blocking states, so
    a tenth state added later defaults to blocking rather than to permitted —
    the direction that fails safe.

    Reporting only. What is actually *withheld* is `refuses_minting`, and what
    lets a campaign move on is `baseline_is_settled`; conflating the three is how
    a closed gate became a campaign that could never open it.
    """
    return state not in ("passed", "waived")


def refuses_minting(state: GateState) -> bool:
    """Whether a belief written now would outlive a baseline nobody trusts.

    Refuses where the campaign can *act*: `failed` is a pipeline worse than a
    constant, `blocked_uncertain` is a guessed target, `stale` is a reading that
    describes a workspace that has moved, and `floor_missing`/`unknown` mean
    nothing has been measured yet.

    It does **not** refuse the unchangeable states. A gate that refuses forever
    on a property of the dataset is one an operator switches off, and then it
    protects nothing at all — which is worse than the wrong beliefs it was
    built to stop.
    """
    return blocks_research(state) and state not in UNCHANGEABLE_STATES


def baseline_is_settled(state: GateState) -> bool:
    """Whether there is anything left for the campaign to do about the baseline.

    `passed` and `waived` because the gate is open; the unchangeable states
    because re-running cannot move them. Everything else is work still to do,
    and the campaign keeps asking for a baseline.
    """
    return not blocks_research(state) or state in UNCHANGEABLE_STATES


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
        _live_answers_fingerprint(root),
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _live_answers_fingerprint(root: Path) -> str:
    """The fingerprint of the answers **on disk now**.

    Not `profile["answers_fingerprint"]`, which is the value the profile was
    *built with* — it only changes when the profile is rebuilt, so reading it
    here could never detect an answer. The real sequence writes
    `schema_answers.json` and leaves `profile.json` untouched until the next
    `prepare_workspace`, and in that window the gate was reporting `passed` over
    a measurement of the column the operator had just rejected.

    `_profile_state` compares this value against the stored one; this is the
    side of that comparison that moves.
    """
    try:
        from labpilot.accessor.profiler.questions import answers_fingerprint, load_answers

        return answers_fingerprint(load_answers(Path(root)))
    except Exception as exc:  # noqa: BLE001 — unreadable answers are not a change
        logger.info("Could not read schema answers: %s", exc)
        return ""


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

    # 2. Is there a plan at all? `floor_missing` says "run the baseline", and
    #    without a `ValidationPlan` there is nothing to run one *under* — advice
    #    the operator cannot follow. `unknown` is the state for that, and it was
    #    unreachable until this check existed.
    if not (root / "baseline_choice.json").is_file():
        verdict.state = "unknown"
        verdict.reason = (
            "no baseline_choice.json, so there is no validation plan to measure a floor under"
        )
        return verdict

    if floor is None:
        verdict.state = "floor_missing"
        verdict.reason = "no baseline_floor.json in this workspace; run the baseline"
        return verdict

    # 3. Did the dataset move under the reading? Asked before what it says,
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


def refuse_hypothesis_minting(workspace_root: Path | None, *, enforced: bool | None = None) -> str:
    """Why hypothesis minting must not proceed here, or empty when it may.

    M23 step 8. **Enforcement is at hypothesis generation, not at submission.**
    A campaign whose pipeline loses to a constant may still run plans, implement
    and reflect — those are how the gate gets *opened* — but it may not mint
    hypotheses, because that is where a false belief enters the store and
    outlives the run. rogii's cost was 19 child hypotheses and eight techniques
    driven to 0.0 confidence, all written down while the pipeline was 91x worse
    than one line of code. None of it was a submission.

    Returns a reason rather than a bool: a refusal an operator cannot act on is
    a wall, and every one of the nine states has a different next step.

    `enforced=None` reads the config, so the flip is one setting rather than a
    code change — and `False` is still the default until a campaign's worth of
    recorded verdicts turns "the gate is right" into a false-positive rate.
    """
    if workspace_root is None:
        # Nothing to judge. A caller with no workspace is not a campaign that has
        # skipped its baseline; it is one this gate cannot see, and refusing on
        # absence would block every path that never had a root to begin with.
        return ""
    if enforced is None:
        enforced = enforcement_enabled()
    if not enforced:
        return ""
    try:
        verdict = evaluate_gate(Path(workspace_root), enforced=True)
    except Exception as exc:  # noqa: BLE001 — a gate that cannot run must not
        # block a campaign. A fault here would read as "baseline not passed",
        # which is the failure mode `H-BASELINE.status` was rejected for.
        logger.warning("Baseline gate could not be evaluated: %s", exc)
        return ""
    if not (verdict.enforced and refuses_minting(verdict.state)):
        return ""
    return f"baseline gate is {verdict.state}: {verdict.reason}"


def enforcement_enabled() -> bool:
    """Whether the rollout has moved past observe-only.

    Read from config on every call rather than captured at import, so flipping it
    does not require a restart — and so a test can flip it without reaching into
    module state.
    """
    try:
        from labpilot.config import load_config

        return bool(getattr(load_config().baseline_gate, "enforced", False))
    except Exception:  # noqa: BLE001 — no config is not enforcement
        return False


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
