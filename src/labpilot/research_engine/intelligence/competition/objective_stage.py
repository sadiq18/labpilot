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
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.competition.direction_probe import Direction
from labpilot.research_engine.intelligence.competition.objective import (
    ObjectiveSpec,
    resolve_objective,
)

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


def read_inputs(root: Path) -> ObjectiveInputs:
    """What this workspace says its objective is, before any resolution.

    Precedence for the metric is profile → contract → harness, and the order is
    the point: the profile's `MetricRef` is what the *dataset understanding
    stage* was given and carried, so reading the contract again here would be a
    second parser of the same file free to drift from the first. The contract is
    the fallback for a workspace whose profile predates M22 or never named one.
    """
    root = Path(root)
    from labpilot.research_engine.validation import harness

    target, profile_metric = _profile_facts(root)
    profile_read = (root / "profile.json").is_file()

    metric_raw: str | None = None
    declared: Direction | None = None
    metric_from = "none"
    if profile_metric:
        raw = profile_metric.get("name") or profile_metric.get("key")
        metric_raw = str(raw) if raw else None
        direction = profile_metric.get("direction")
        declared = direction if direction in ("maximize", "minimize") else None
        if metric_raw:
            metric_from = "profile"
    if not metric_raw:
        metric_raw, declared = _contract_metric(root)
        if metric_raw:
            metric_from = "competition.json"
    if not metric_raw and harness.handles(root):
        # A workspace with no competition contract is not a workspace with no
        # objective. A benchmark states its own, and refusing it here would be
        # the domain leak the preflight already removed once.
        metric_raw, stated = harness.stated_objective(root)
        declared = stated if stated in ("maximize", "minimize") else None
        if metric_raw:
            metric_from = harness.OBJECTIVE_FILE

    return ObjectiveInputs(
        target=target,
        task=_contract_task(root),
        metric_raw=metric_raw,
        declared_direction=declared,
        externally_scored=harness.handles(root),
        metric_from=metric_from,
        profile_read=profile_read,
    )


def resolve_workspace_objective(root: Path, competition: str = "") -> StoredObjective:
    """Resolve from the workspace. Always returns; never raises on a bad input.

    A workspace that states nothing resolves to an objective that says so —
    `unresolved: ["metric"]` — which is a fact worth writing down. Returning
    `None` here would push "we do not know" back into the caller's absence
    handling, where it reads as "not asked yet".
    """
    inputs = read_inputs(root)
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


def write_objective(root: Path, stored: StoredObjective) -> Path:
    """Write `objective.json`, stamping the schema version here.

    Stamped by the one writer rather than defaulted on the model, so the field
    says which resolver produced *this file* — the same reason `write_profile`
    does it, and the same defect avoided: a default makes every unstamped legacy
    file validate as current.
    """
    path = Path(root) / OBJECTIVE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
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


def objective_state(root: Path) -> str:
    """``missing``, ``unusable``, ``stale``, or ``current``.

    ``stale`` and ``unusable`` are kept apart for the reason `_profile_state`
    keeps them apart: one is a file worth re-resolving from a workspace that has
    since changed, the other is bytes no reader can use.
    """
    path = Path(root) / OBJECTIVE_FILENAME
    if not path.is_file():
        return "missing"
    stored = load_objective(root)
    if stored is None:
        return "unusable"
    if stored.schema_version != OBJECTIVE_SCHEMA_VERSION:
        return "stale"
    if stored.inputs != read_inputs(root):
        return "stale"
    return "current"


def ensure_objective(root: Path, competition: str = "") -> tuple[StoredObjective, str]:
    """The objective for this workspace, and how it was obtained.

    Returns `(stored, state)` where state is ``reused`` or ``resolved``. Reuse
    matters less here than it does for a profile — resolution is cheap — but a
    *stable* answer matters a great deal: a campaign whose objective is
    re-derived on every command can change its mind halfway through without
    anything recording that it did.
    """
    root = Path(root)
    if objective_state(root) == "current":
        stored = load_objective(root)
        if stored is not None:
            return stored, "reused"
    stored = resolve_workspace_objective(root, competition)
    write_objective(root, stored)
    return stored, "resolved"
