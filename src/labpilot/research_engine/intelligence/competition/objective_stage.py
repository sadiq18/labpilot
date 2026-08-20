"""The objective as a stage: read from the schema, written down, reused.

M23 step 0. `resolve_objective` has been right since #145 and reached nobody:
it was called once, from a CLI preflight, and its answer was flattened to five
strings on session metadata. The two fields that matter most —
`contradiction`, naming two sources that disagree, and `unresolved`, naming
exactly what to ask about — reached a console line and then nothing.

So this module does three things and no more:

* **Reads the inputs from the workspace**, not from CLI arguments. The target
  comes from `profile.json`, which is the stage before; the metric comes from
  the profile's own `MetricRef` when it has one, and from the contract when it
  does not. A stage that re-derives what the stage before it already resolved is
  not a pipeline, it is two implementations of the same question.
* **Writes `objective.json` beside `profile.json`**, so every later stage reads
  one answer instead of re-resolving from the profile. Validation strategy is
  the first to do so (step 2), and `selector.py` mentions no objective at all
  today.
* **Reuses it only while its inputs still hold.** The inputs are *stored*, and
  staleness is the comparison — not a fingerprint recorded beside them, which
  would be a second copy of the same fact free to disagree with the first.

Nothing here decides whether a campaign may run. `ObjectiveSpec.blocks_launch`
already answers that, the CLI preflight already consults it, and the gate that
enforces it is step 8.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.competition.direction_probe import Direction
from labpilot.research_engine.intelligence.competition.objective import (
    ObjectiveSpec,
    resolve_objective,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OBJECTIVE_FILENAME",
    "OBJECTIVE_SCHEMA_VERSION",
    "ObjectiveInputs",
    "StoredObjective",
    "ensure_objective",
    "load_objective",
    "objective_state",
    "read_inputs",
    "resolve_workspace_objective",
    "write_objective",
]

OBJECTIVE_FILENAME = "objective.json"

#: Bumped when a stored objective can no longer be read as this version means
#: it. A version bump re-resolves; it never migrates, because re-resolving is
#: cheap and a migration would be code asserting what an older resolver meant.
OBJECTIVE_SCHEMA_VERSION = 1


class ObjectiveInputs(BaseModel):
    """Everything the resolution depended on, so reuse can be justified.

    Stored rather than hashed: a fingerprint says *that* something changed, and
    these say *what*, which is the difference between "re-resolving" and an
    operator being able to see why. Comparison is by value — pydantic models
    already do that — so there is no second copy of the same fact.
    """

    #: From `profile.json`. The whole reason this is a stage and not a helper:
    #: the target is M22's answer, and the objective now reads it rather than
    #: re-deriving it.
    target: str | None = None
    #: Today the contract's `problem_type`. Step 1 replaces this with the
    #: schema's measured `target_type`, and the swap is one line here.
    task: str | None = None
    metric_raw: str | None = None
    declared_direction: Direction | None = None
    #: A benchmark reports its own score, so "nothing here can compute it" is
    #: true and irrelevant. Carried because it changes the resolution.
    externally_scored: bool = False
    #: Where the metric came from, for a human reading the file. Not a decision
    #: input — `metric_raw` is — but the difference between a contract and a
    #: profile is the first thing anyone asks when the two disagree.
    metric_from: str = "none"
    #: Whether a profile was read at all. A `target: null` resolved from no
    #: profile and one resolved from a profile that could not name a target are
    #: different situations with the same value.
    profile_read: bool = False


class StoredObjective(BaseModel):
    """`objective.json`: what was resolved, and what it was resolved from."""

    schema_version: int = 0
    competition: str = ""
    inputs: ObjectiveInputs = Field(default_factory=ObjectiveInputs)
    spec: ObjectiveSpec = Field(default_factory=ObjectiveSpec)


def _profile_facts(root: Path) -> tuple[str | None, dict[str, Any] | None]:
    """`(target_column, metric)` from `profile.json`, or `(None, None)`.

    Unreadable is the same as absent on purpose. A corrupt profile is a problem
    for the profile stage to report; refusing to resolve an objective over it
    would turn one broken file into two.
    """
    path = Path(root) / "profile.json"
    if not path.is_file():
        return None, None
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(profile, dict):
        return None, None
    target = profile.get("target_column")
    metric = profile.get("metric")
    return (str(target) if target else None), (metric if isinstance(metric, dict) else None)


def _contract_metric(root: Path) -> tuple[str | None, Direction | None]:
    """`(name, direction)` from `competition.json`.

    Both key spellings, because hand-written contracts use `metric` and the
    parser writes `evaluation_metric` — `direction.py` reads both for the same
    reason.
    """
    path = Path(root) / "competition.json"
    if not path.is_file():
        return None, None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(spec, dict):
        return None, None
    metric = spec.get("evaluation_metric") or spec.get("metric") or {}
    if not isinstance(metric, dict):
        return None, None
    raw = metric.get("name") or metric.get("key")
    declared = metric.get("direction")
    return (
        str(raw) if raw else None,
        declared if declared in ("maximize", "minimize") else None,
    )


def _contract_task(root: Path) -> str | None:
    path = Path(root) / "competition.json"
    if not path.is_file():
        return None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(spec, dict):
        return None
    return str(spec.get("problem_type") or "") or None


def _profile_metric(metric: dict[str, Any] | None) -> tuple[str | None, Direction | None]:
    """`(name, direction)` from the profile's `MetricRef`."""
    if not metric:
        return None, None
    raw = metric.get("name") or metric.get("key")
    direction = metric.get("direction")
    return (
        str(raw) if raw else None,
        direction if direction in ("maximize", "minimize") else None,
    )


def _harness_metric(root: Path) -> tuple[str | None, Direction | None]:
    """`(name, direction)` a harness promises.

    Read whenever `harness.json` is there and nothing above it stated a metric —
    *not* only when `harness.handles()` is true. `handles()` also answers "is
    this a harness workspace rather than a Kaggle one", and it is false as soon
    as a `metrics.json` appears beside the declaration. Gating the *metric* on it
    meant a benchmark workspace lost its objective the moment it produced a
    result: it launched before its first run and was refused after, with advice
    naming a `competition.json` it will never have.
    """
    from labpilot.research_engine.validation import harness

    if not (Path(root) / harness.OBJECTIVE_FILE).is_file():
        return None, None
    metric, direction = harness.stated_objective(Path(root))
    return metric, direction if direction in ("maximize", "minimize") else None


def read_inputs(root: Path) -> ObjectiveInputs:
    """What this workspace says its objective is, before any resolution.

    Precedence for the metric's *identity* is profile → contract → harness, and
    the order is the point: the profile's `MetricRef` is what the dataset
    understanding stage was given and carried, so reading the contract again
    would be a second parser of the same file free to drift from the first.

    **Direction is resolved separately, down the same chain.** Letting whichever
    source won the identity also claim the direction slot dropped a declaration
    the operator had just made: the profiler writes `direction: null` for any
    contract whose direction is absent or invalid, so a profile naming the metric
    shadowed the `harness.json` or `competition.json` that oriented it — and the
    campaign was refused for an unknown direction that was stated in the
    workspace, in a file the refusal then told the operator to go and edit.

    A later source may only orient the metric that won, so a contract naming
    `rmse` never lends its `minimize` to a profile naming something else.
    """
    root = Path(root)
    from labpilot.research_engine.intelligence.competition.metric_vocabulary import _slug
    from labpilot.research_engine.validation import harness

    target, profile_metric = _profile_facts(root)
    claims: list[tuple[str, str | None, Direction | None]] = [
        ("profile", *_profile_metric(profile_metric)),
        ("competition.json", *_contract_metric(root)),
        (harness.OBJECTIVE_FILE, *_harness_metric(root)),
    ]

    metric_raw: str | None = None
    metric_from = "none"
    for name, raw, _direction in claims:
        if raw:
            metric_raw, metric_from = raw, name
            break

    declared: Direction | None = None
    for _name, raw, direction in claims:
        if direction is None:
            continue
        # A source that names no metric is talking about whichever one won;
        # one that names a different metric is talking about something else.
        if raw is None or (metric_raw is not None and _slug(raw) == _slug(metric_raw)):
            declared = direction
            break

    return ObjectiveInputs(
        target=target,
        task=_contract_task(root),
        metric_raw=metric_raw,
        declared_direction=declared,
        externally_scored=harness.handles(root),
        metric_from=metric_from,
        profile_read=(root / "profile.json").is_file(),
    )


def _resolve(inputs: ObjectiveInputs, competition: str) -> StoredObjective:
    """Resolve from inputs that have already been read.

    Split out so `ensure_objective` reads the workspace once. Asking
    `objective_state` and then re-resolving parsed `profile.json`,
    `competition.json` and `harness.json` twice per call and ran the direction
    probe against the second reading — the same duplication `_profile_state`
    carries a comment about.
    """
    spec = resolve_objective(
        metric_raw=inputs.metric_raw,
        declared_direction=inputs.declared_direction,
        task=inputs.task,
        target=inputs.target,
        externally_scored=inputs.externally_scored,
    )
    return StoredObjective(
        schema_version=OBJECTIVE_SCHEMA_VERSION,
        competition=competition,
        inputs=inputs,
        spec=spec,
    )


def resolve_workspace_objective(root: Path, competition: str = "") -> StoredObjective:
    """Resolve from the workspace. Always returns; never raises on a bad input.

    A workspace that states nothing resolves to an objective that says so —
    `unresolved: ["metric"]` — which is a fact worth writing down. Returning
    `None` here would push "we do not know" back into the caller's absence
    handling, where it reads as "not asked yet".
    """
    return _resolve(read_inputs(root), competition)


def write_objective(root: Path, stored: StoredObjective) -> Path:
    """Write `objective.json`, stamping the schema version here.

    Stamped by the one writer rather than defaulted on the model, so the field
    says which resolver produced *this file* — the same reason `write_profile`
    does it, and the same defect avoided: a default makes every unstamped legacy
    file validate as current.

    Deliberately does **not** create the workspace. Resolving an objective is a
    read of a workspace that already exists, and `mkdir(parents=True)` here meant
    the launch preflight materialised `competitions/<typo>/objective.json` on its
    way to refusing the campaign — a check with a side effect, and the side
    effect was a workspace nobody asked for.
    """
    path = Path(root) / OBJECTIVE_FILENAME
    stamped = stored.model_copy(update={"schema_version": OBJECTIVE_SCHEMA_VERSION})
    path.write_text(stamped.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_objective(root: Path) -> StoredObjective | None:
    """The stored objective, or `None` when there is nothing readable there."""
    path = Path(root) / OBJECTIVE_FILENAME
    if not path.is_file():
        return None
    try:
        return StoredObjective.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _state_of(root: Path, stored: StoredObjective | None, inputs: ObjectiveInputs) -> str:
    """The verdict, over a file and inputs the caller has already read."""
    if not (Path(root) / OBJECTIVE_FILENAME).is_file():
        return "missing"
    if stored is None:
        return "unusable"
    if stored.schema_version != OBJECTIVE_SCHEMA_VERSION:
        return "stale"
    if stored.inputs != inputs:
        return "stale"
    return "current"


def objective_state(root: Path) -> str:
    """``missing``, ``unusable``, ``stale``, or ``current``.

    ``stale`` and ``unusable`` are kept apart for the reason `_profile_state`
    keeps them apart: one is a file worth re-resolving from a workspace that has
    since changed, the other is bytes no reader can use.
    """
    root = Path(root)
    return _state_of(root, load_objective(root), read_inputs(root))


def ensure_objective(root: Path, competition: str = "") -> tuple[StoredObjective, str]:
    """The objective for this workspace, and how it was obtained.

    Returns `(stored, how)` where `how` is ``reused``, ``resolved``, or
    ``unpersisted``. Reuse matters less here than it does for a profile —
    resolution is cheap — but a *stable* answer matters a great deal: a campaign
    whose objective is re-derived on every command can change its mind halfway
    through without anything recording that it did.

    ``unpersisted`` is the honest third answer, and it exists because this is
    called from the launch preflight. That preflight used to be a pure read; a
    write that raises turns a campaign that would have started into a
    `PermissionError` traceback, and a resolved objective is worth having even
    when the workspace cannot hold it. The caller is told, rather than the
    failure being swallowed.
    """
    root = Path(root)
    inputs = read_inputs(root)
    stored = load_objective(root)
    if _state_of(root, stored, inputs) == "current" and stored is not None:
        return stored, "reused"

    stored = _resolve(inputs, competition)
    try:
        write_objective(root, stored)
    except OSError as exc:
        # One guard, not two. An `is_dir()` check in front of this caught the
        # absent-workspace case that the writer's own `FileNotFoundError`
        # already lands here, and no test could tell the two apart — a branch
        # nothing distinguishes is a branch nothing maintains.
        logger.warning("Could not persist %s in %s: %s", OBJECTIVE_FILENAME, root, exc)
        return stored, "unpersisted"
    return stored, "resolved"
