"""Produce the two readings, once, and hand them to whoever asks.

M23 step 7. Until now nothing in `src/` called `compute_floor` or
`fit_baseline_one` — both were reachable only from tests, so no campaign ever
wrote `baseline_floor.json` and the gate would have reported `floor_missing`
forever. This is the caller that was missing.

**Load before compute.** The readings are expensive relative to everything else
on the COMPARE path (five LightGBM fits), and a campaign that re-measured its own
control on every read would be spending real time to get the same number — and
would have no way to notice if it did not.

**A failure is a reading that says why, never an exception.** The gate has nine
states and none of them is "the baseline crashed"; a workspace this cannot
measure gets `floor_undefined` or `awaiting_ml`, which are answers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from labpilot.research_engine.execution.baseline.baseline_one import (
    ModelReading,
    fit_baseline_one,
    load_baseline_one,
    write_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import (
    FloorReading,
    floor_for_workspace,
    load_floor,
    write_floor,
)
from labpilot.research_engine.execution.baseline.gate import reading_fingerprint
from labpilot.research_engine.execution.baseline.selector import ValidationPlan

logger = logging.getLogger(__name__)

__all__ = ["ensure_readings", "floor_as_control"]


def _read(root: Path, name: str) -> dict:
    path = Path(root) / name
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _current(reading: FloorReading | ModelReading | None, fingerprint: str) -> bool:
    """Whether a reading on disk still describes this workspace.

    An unstamped reading counts as current, the same as in the gate: every one
    written before that field existed has an empty stamp, and re-measuring every
    workspace on upgrade would teach an operator to ignore the distinction.
    """
    if reading is None:
        return False
    return not (reading.workspace_fingerprint and reading.workspace_fingerprint != fingerprint)


def ensure_readings(root: Path) -> tuple[FloorReading | None, ModelReading | None]:
    """The floor and the model for this workspace, computed if they are not there.

    Both are stamped with the gate's fingerprint at the moment they are taken, so
    a later answer or re-profile makes them stale rather than silently wrong.
    """
    root = Path(root)
    fingerprint = reading_fingerprint(root)

    floor = load_floor(root)
    if not _current(floor, fingerprint):
        try:
            floor = floor_for_workspace(root)
            # Stamped here and nowhere else. `floor_for_workspace` used to do it
            # too, which made two writers for one field — and forced a late
            # import of the gate from `floor.py` purely to dodge the cycle that
            # duplication created. A mutation sweep found it: removing the stamp
            # from one of the two changed nothing, which is what redundant means.
            floor.workspace_fingerprint = fingerprint
            write_floor(root, floor)
        except Exception as exc:  # noqa: BLE001 — a floor that cannot be taken
            # is not a campaign that should stop; the gate has a state for it.
            logger.info("Could not compute a floor for %s: %s", root, exc)
            return None, None

    model = load_baseline_one(root)
    if not _current(model, fingerprint):
        model = _fit(root, fingerprint)
        if model is not None:
            write_baseline_one(root, model)
    return floor, model


def _fit(root: Path, fingerprint: str) -> ModelReading | None:
    """Baseline 1 over the same table and plan the floor used."""
    choice, profile = _read(root, "baseline_choice.json"), _read(root, "profile.json")
    metric_name = str(choice.get("metric_name") or "")
    target = str(choice.get("target_column") or profile.get("target_column") or "")
    train_file = choice.get("train_file") or profile.get("train_file")
    if not (metric_name and target and train_file):
        return None
    candidates = [
        Path(str(train_file)),
        root / str(train_file),
        root / "data" / "raw" / str(train_file),
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        logger.info("Could not read %s for Baseline 1: %s", path, exc)
        return None

    # `feature_columns`, `target_type` and `modality` are computed fields, so
    # they are already *in* the file. Read from the dict rather than validating
    # the whole `DatasetProfile`: a single malformed corner — a legacy
    # `modalities` entry missing `role` was enough — made validation raise, and
    # `_fit` then returned None, which the gate reported as `awaiting_ml`.
    #
    # That conflation matters now that `awaiting_ml` is one of the states a
    # campaign is allowed to move past: an unparseable profile would have read
    # as "Baseline 1 is merely unaffordable here" and waved the campaign
    # through. Recomputing these would be a fourth implementation of questions
    # the schema has settled with evidence; reading them from the file is not.
    target_type, modality, features, class_counts = _schema_answers(profile, frame, target)

    reading = fit_baseline_one(
        frame,
        target=target,
        plan=ValidationPlan.model_validate(choice.get("validation") or {}),
        metric_name=metric_name,
        target_type=target_type,
        feature_columns=features or [c for c in frame.columns if c != target],
        modality=modality,
        num_classes=len(class_counts) if class_counts else None,
    )
    reading.workspace_fingerprint = fingerprint
    return reading


def _schema_answers(
    profile: dict, frame: pd.DataFrame, target: str
) -> tuple[str, str, list[str], dict]:
    """`(target_type, modality, feature_columns, class_counts)` from the schema.

    Two readings of one file, in this order for a reason.

    The **model** first, because `target_type`, `modality` and `feature_columns`
    are computed fields: a profile written by an older profiler does not have
    them in the JSON at all, and validating is what derives them. Recomputing
    them here would be a fourth implementation of questions the schema has
    already settled with evidence.

    The **raw dict** as the fallback, because validation is all-or-nothing: one
    malformed corner — a legacy `modalities` entry missing `role` was enough —
    made the whole thing raise, and `_fit` then returned None, which the gate
    reports as `awaiting_ml`. That conflation matters now that `awaiting_ml` is a
    state a campaign may move past: an unparseable profile would read as
    "Baseline 1 is merely unaffordable here" and wave the campaign through.
    """
    from labpilot.accessor.profiler.tabular import DatasetProfile

    def _usable(columns: object) -> list[str]:
        return [
            c
            for c in (columns or [])  # type: ignore[union-attr]
            if isinstance(c, str) and c in frame.columns and c != target
        ]

    try:
        described = DatasetProfile.model_validate(profile)
    except ValueError as exc:
        logger.info("Profile did not validate, reading its stored fields instead: %s", exc)
        distribution = profile.get("target_distribution")
        return (
            str(profile.get("target_type") or "unknown"),
            str(profile.get("modality") or "tabular"),
            _usable(profile.get("feature_columns")),
            distribution.get("class_counts") or {} if isinstance(distribution, dict) else {},
        )
    return (
        described.target_type,
        described.modality,
        _usable(described.feature_columns),
        described.target_distribution.class_counts,
    )


def floor_as_control(floor: FloorReading | None) -> dict[str, float]:
    """`{"cv_<metric>": score}` — a reading already taken, shaped as a control.

    Takes the reading rather than a root, so producing it is a separate and
    visible step. The previous version called `ensure_readings` itself, which
    meant a name promising a dict lookup ran five LightGBM fits inside
    `resolve_control` — measured at 1.0s on 5,000x30, against a cell budget
    permitting 133x that.

    This is §7.5's whole design, and the restraint is the point: **no third
    reading on `ObservedOutcomes`**. `_decide` is the single funnel for every
    verdict in the system, so the floor arrives as `parent_cv` and everything
    downstream — the gain, the sign, the card, `H-BASELINE`'s status — works
    unchanged. A metric mismatch is then caught for free by `_same_metric`,
    machinery that already exists and has already been debugged.
    """
    if floor is None or not floor.is_defined or floor.score is None:
        return {}
    return {f"cv_{floor.metric_name}": float(floor.score)}
