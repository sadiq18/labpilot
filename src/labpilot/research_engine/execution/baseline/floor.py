"""What the dumbest defensible answer scores. A fact about the dataset.

M23 step 3. A model that cannot beat a constant has not learned anything, and
until now nothing in this system could say so — `_observe_delta` compared runs
against each other, so a campaign could improve steadily while sitting below the
score you get by predicting the mean.

Four rules, and each one is a way this measurement can be made worthless:

* **The folds are the model's own.** Read from `baseline_choice.json`, never
  re-derived here. A floor computed on a different split is not a floor, it is a
  second number that happens to be in the same units.
* **Fitted per fold, on the train side only.** Fitting the constant on the whole
  target is the leakage version: it looks unbeatable on skewed data, which is
  exactly where a floor most needs to be honest.
* **Every strategy is recorded, and the floor is the best of them.** A gate that
  picked the worse constant is a gate too easy to pass — and the losing
  strategies are what make a suspiciously low floor legible.
* **Scored by `compute_metric`.** One metric implementation, never a second, so
  the floor and the model's `cv_<metric>` are the same number computed the same
  way. That is what makes the comparison free.

Plan: ``docs/research-os/autonomy-roadmap/design/18-baseline-correctness.md`` §7.1
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from labpilot.research_engine.execution.baseline.selector import ValidationPlan
from labpilot.research_engine.execution.metrics import compute_metric

logger = logging.getLogger(__name__)

__all__ = [
    "FLOOR_FILENAME",
    "fingerprint_of",
    "folds_for",
    "FloorReading",
    "compute_floor",
    "floor_for_workspace",
    "load_floor",
    "write_floor",
]

FLOOR_FILENAME = "baseline_floor.json"

#: Which constants are worth trying, per metric. The optimal one follows the
#: metric, not the target's dtype: mean minimises squared error and median
#: minimises absolute error over the *same* column, and a floor that only knew
#: one of them would be the wrong constant for half the competitions.
#:
#: More than one is listed on purpose. The floor is the best of them, and the
#: ones that lost are what tell a reader whether a low floor means "the target is
#: easy" or "this strategy was a poor fit".
_STRATEGIES_BY_METRIC: dict[str, tuple[str, ...]] = {
    "rmse": ("mean", "median"),
    "root_mean_squared_error": ("mean", "median"),
    "mse": ("mean", "median"),
    "mae": ("median", "mean"),
    "rmsle": ("log_mean", "median", "mean"),
    "accuracy": ("majority_class",),
    "acc": ("majority_class",),
    "f1": ("majority_class",),
    "logloss": ("class_prior",),
    "log_loss": ("class_prior",),
    # AUC is not computed at all — see `_ANALYTIC`.
    "auc": (),
    "roc_auc": (),
    "roc-auc": (),
}

#: Metrics whose floor is a theorem rather than a measurement. A constant
#: prediction carries no ranking information, so its ROC AUC is exactly 0.5 for
#: every dataset. Computing it invites a fold with one class present to return
#: NaN, or 0.0, and a floor of 0.0 is one every model clears.
_ANALYTIC: dict[str, float] = {"auc": 0.5, "roc_auc": 0.5, "roc-auc": 0.5}


class FloorReading(BaseModel):
    """A dataset reading, deliberately not an experiment result.

    No `hypothesis_id`, no execution id, no generated file: this is a property of
    the data under a stated split, and anything that made it look like a run
    would invite it to be compared against runs as though it were one.
    """

    metric_name: str
    #: Every strategy tried, and what it scored. The winner is `best_strategy`.
    strategies: dict[str, float] = Field(default_factory=dict)
    best_strategy: str = ""
    score: float | None = None
    #: The plan this was computed under, copied rather than referenced — a
    #: reading whose split you cannot see is not checkable.
    validation: ValidationPlan = Field(default_factory=ValidationPlan)
    #: Over the target's bytes, the plan and the metric. A re-profiled dataset or
    #: a changed answer moves it, which is how a stale floor is spotted.
    fingerprint: str = ""
    computed_at: str = ""
    #: Why there is no floor, when there is none. An empty `strategies` with no
    #: reason cannot be told apart from a computation that was never attempted,
    #: and the gate's `floor_undefined` state needs the difference.
    undefined_reason: str = ""

    @property
    def is_defined(self) -> bool:
        return self.score is not None and not self.undefined_reason


# --- the strategies ----------------------------------------------------------


def _constant_for(strategy: str, y_train: pd.Series) -> float | object | None:
    """The constant this strategy would predict, fitted on the train side."""
    values = y_train.dropna()
    if values.empty:
        return None
    if strategy == "mean":
        return float(values.mean())
    if strategy == "median":
        return float(values.median())
    if strategy == "log_mean":
        # The optimal constant under RMSLE: the error is measured in log space,
        # so the constant that minimises it is the mean *there*, carried back.
        # The arithmetic mean is not it, and on a skewed price target the gap is
        # the difference between a floor a model clears and one it does not.
        if bool((values < 0).any()):
            return None
        return float(np.expm1(np.log1p(values.astype(float)).mean()))
    if strategy == "majority_class":
        counts = values.value_counts()
        if counts.empty:
            return None
        # `sort_index` first so ties break on the label rather than on the order
        # pandas happened to see them, which makes the floor reproducible.
        top = counts[counts == counts.max()].sort_index()
        return top.index[0]
    if strategy == "class_prior":
        return None  # handled as probabilities, not a point prediction
    raise ValueError(f"unknown floor strategy: {strategy}")


def _predict(
    strategy: str,
    y_train: pd.Series,
    val_frame: pd.DataFrame,
    *,
    anchor_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """`(y_pred, y_proba)` for the validation rows, or None if inapplicable."""
    import pandas as pd

    n = len(val_frame)
    if strategy == "anchor_carry_forward":
        if not anchor_column or anchor_column not in val_frame.columns:
            return None
        carried = pd.to_numeric(val_frame[anchor_column], errors="coerce")
        if carried.isna().all():
            return None
        # Forward-fill, then fall back to the train side's last known value for
        # any leading gap — a carried value is only a prediction where there is
        # something to carry.
        filled = carried.ffill()
        if filled.isna().any():
            fallback = pd.to_numeric(y_train, errors="coerce").dropna()
            if fallback.empty:
                return None
            filled = filled.fillna(float(fallback.iloc[-1]))
        return filled.to_numpy(dtype=float), None

    if strategy == "class_prior":
        values = y_train.dropna()
        if values.empty:
            return None
        prior = values.value_counts(normalize=True).sort_index()
        if len(prior) != 2:
            # `compute_metric` degrades multiclass log-loss to accuracy, which
            # would silently score a different quantity. Say it does not apply.
            return None
        positive = float(prior.iloc[-1])
        y_proba = np.full(n, positive, dtype=float)
        y_pred = np.full(n, prior.index[-1] if positive >= 0.5 else prior.index[0])
        return y_pred, y_proba

    constant = _constant_for(strategy, y_train)
    if constant is None:
        return None
    return np.full(n, constant), None


# --- the folds ---------------------------------------------------------------


def _build_folds(plan: ValidationPlan, frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """The model's own splits, or an empty list when the plan cannot be honoured.

    Public because Baseline 1 must split identically — three numbers on three
    splits compare nothing — and a private name reached across a module boundary
    is a contract nothing states.

    Empty rather than a silent fallback to `KFold`: the plan is the whole point,
    and a floor computed on a split the model will not use is a number in the
    right units and the wrong universe.
    """
    n = len(frame)
    n_splits = max(int(plan.n_splits or 5), 2)
    if n < n_splits:
        return []
    index = np.arange(n)

    if plan.scheme == "group_kfold":
        key = plan.group_key
        if not key or key not in frame.columns:
            return []
        from sklearn.model_selection import GroupKFold

        groups = frame[key].astype(str).to_numpy()
        if len(np.unique(groups)) < n_splits:
            return []
        return [(tr, va) for tr, va in GroupKFold(n_splits=n_splits).split(index, groups=groups)]

    if plan.scheme == "partition_suffix_holdout":
        # Each partition's tail is held out, reproducing the predict-forward gap
        # the competition scores. Without a group key there are no partitions to
        # take a tail of.
        key = plan.group_key
        if not key or key not in frame.columns:
            return []
        fraction = float(plan.holdout_fraction or 0.5)
        if not 0.0 < fraction < 1.0:
            return []
        validation: list[int] = []
        # `.indices` returns *positions*; the previous version round-tripped
        # through index labels via `get_indexer`, which requires a unique index —
        # and a frame concatenated from per-partition files keeps each file's own
        # 0..n, so the one scheme written for rogii raised `InvalidIndexError` on
        # rogii's own layout.
        for _, positions in sorted(frame.groupby(frame[key].astype(str)).indices.items()):
            ordered = np.sort(np.asarray(positions, dtype=int))
            cut = max(1, int(round(len(ordered) * fraction)))
            validation.extend(ordered[-cut:].tolist())
        val = np.array(sorted(set(validation)), dtype=int)
        train = np.array([i for i in index if i not in set(val.tolist())], dtype=int)
        if train.size == 0 or val.size == 0:
            return []
        return [(train, val)]

    from sklearn.model_selection import KFold

    # `shuffle=False`: the same bytes and the same plan must give the same floor,
    # and a shuffle would need a seed that `ValidationPlan` does not carry.
    return [(tr, va) for tr, va in KFold(n_splits=n_splits, shuffle=False).split(index)]


def fingerprint_of(y: pd.Series, plan: ValidationPlan, metric_name: str) -> str:
    """A digest of the target's **values**, the plan, and the metric.

    `hash_pandas_object` and not `to_numpy().tobytes()`: for an object dtype the
    latter hashes Python object *addresses*, so two processes reading the same
    CSV produced different digests for the same data. Every string-labelled
    classification competition would then have reported `stale` on every read —
    and a staleness state that always fires is one everybody learns to ignore.

    Exported rather than private because Baseline 1 must produce the same digest
    over the same target; two implementations of one fingerprint is how the floor
    and the model come to disagree about whether they described the same data.
    """
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(y, index=False).to_numpy().tobytes())
    digest.update(plan.model_dump_json().encode("utf-8"))
    digest.update(metric_name.encode("utf-8"))
    return digest.hexdigest()


# --- the reading -------------------------------------------------------------


def folds_for(plan: ValidationPlan, frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """The model's own splits. Returns `[]` rather than raising, always.

    One public entry with one contract, used by the floor and by Baseline 1 —
    three numbers on three splits compare nothing, and a private name reached
    across a module boundary is a contract nothing states.

    Empty covers both "this plan does not apply here" and "applying it failed":
    every other step in this module reports an `undefined_reason`, and splitting
    was the one that could still take a caller down with a pandas exception.
    "The gate crashed" is not one of the nine states.
    """
    try:
        return _build_folds(plan, frame)
    except Exception as exc:  # noqa: BLE001 — any split failure is "no folds"
        logger.info("Validation plan %r could not be applied: %s", plan.scheme, exc)
        return []


def compute_floor(
    frame: pd.DataFrame,
    *,
    target: str,
    plan: ValidationPlan,
    metric_name: str,
    direction: str,
    anchor_column: str | None = None,
    num_classes: int | None = None,
) -> FloorReading:
    """Score every applicable constant under `plan`, and return the best.

    `direction` decides what "best" means and has no default on purpose: a floor
    picked with the wrong sign is the most convincing wrong number this system
    could produce, and defaulting to `maximize` is how `build_evidence_card` came
    to record rogii's only real improvement as `rejected`.
    """
    now = datetime.now(UTC).isoformat()
    if direction not in ("maximize", "minimize"):
        return FloorReading(
            metric_name=metric_name,
            validation=plan,
            computed_at=now,
            undefined_reason=f"direction {direction!r} is not maximize or minimize",
        )
    if target not in frame.columns:
        return FloorReading(
            metric_name=metric_name,
            validation=plan,
            computed_at=now,
            undefined_reason=(
                f"the profile names {target!r} and the training table has no such column"
            ),
        )

    y = frame[target]
    reading = FloorReading(
        metric_name=metric_name,
        validation=plan,
        fingerprint=fingerprint_of(y, plan, metric_name),
        computed_at=now,
    )

    key = metric_name.lower()
    if key in _ANALYTIC:
        # A theorem, not a measurement. Recorded as a strategy so the artifact
        # reads the same shape either way.
        reading.strategies = {"constant_prediction": _ANALYTIC[key]}
        reading.best_strategy = "constant_prediction"
        reading.score = _ANALYTIC[key]
        return reading

    names = list(_STRATEGIES_BY_METRIC.get(key, ()))
    if not names:
        reading.undefined_reason = f"no floor strategy is defined for metric {metric_name!r}"
        return reading
    if anchor_column:
        # Not a constant at all, and on rogii it is the whole story: the target's
        # known prefix carried forward scores 15.1 against the pipeline's 1380.
        # The profiler has named it since 2026-08-13 with nothing reading it.
        names.append("anchor_carry_forward")

    folds = folds_for(plan, frame)
    if not folds:
        reading.undefined_reason = (
            f"the {plan.scheme!r} plan could not be honoured on this table "
            f"({len(frame)} rows, group key {plan.group_key!r})"
        )
        return reading

    for name in names:
        scores: list[float] = []
        for train_idx, val_idx in folds:
            y_train = y.iloc[train_idx]
            val_frame = frame.iloc[val_idx]
            y_true = val_frame[target]
            predicted = _predict(name, y_train, val_frame, anchor_column=anchor_column)
            if predicted is None:
                scores = []
                break
            y_pred, y_proba = predicted
            usable = y_true.notna().to_numpy()
            if not usable.any():
                scores = []
                break
            try:
                scores.append(
                    compute_metric(
                        y_true[usable].to_numpy(),
                        y_pred[usable],
                        metric_name,
                        y_proba=None if y_proba is None else y_proba[usable],
                        num_classes=num_classes,
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.info("Floor strategy %r not scorable for %s: %s", name, metric_name, exc)
                scores = []
                break
        if scores:
            reading.strategies[name] = float(np.mean(scores))

    if not reading.strategies:
        reading.undefined_reason = "no strategy could be scored against this target"
        return reading

    pick = max if direction == "maximize" else min
    reading.best_strategy = pick(reading.strategies, key=lambda k: reading.strategies[k])
    reading.score = reading.strategies[reading.best_strategy]
    return reading


def floor_for_workspace(root: Path) -> FloorReading:
    """Assemble the inputs from the workspace and compute the floor.

    Everything comes from an artifact a stage before this one wrote: the plan and
    the metric from `baseline_choice.json`, the target and the anchor from
    `profile.json`, the direction from the metric registry. Nothing is re-derived
    here, which is the same rule step 2 applied one layer up — a floor that
    decided its own metric would not be measuring the model's objective.
    """
    root = Path(root)
    now = datetime.now(UTC).isoformat()

    def undefined(reason: str) -> FloorReading:
        return FloorReading(metric_name="", computed_at=now, undefined_reason=reason)

    choice_path = root / "baseline_choice.json"
    profile_path = root / "profile.json"
    if not choice_path.is_file():
        return undefined("no baseline_choice.json, so there is no plan to compute a floor under")
    try:
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        profile = (
            json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
        )
    except (OSError, ValueError) as exc:
        return undefined(f"could not read the workspace artifacts: {exc}")
    if not isinstance(choice, dict) or not isinstance(profile, dict):
        return undefined("baseline_choice.json or profile.json is not an object")

    metric_name = str(choice.get("metric_name") or "")
    target = str(choice.get("target_column") or profile.get("target_column") or "")
    if not metric_name or not target:
        return undefined("the baseline choice names no metric or no target")

    from labpilot.research_engine.intelligence.competition.metric_vocabulary import direction_of

    direction = direction_of(metric_name) or ""
    plan = ValidationPlan.model_validate(choice.get("validation") or {})

    train_file = choice.get("train_file") or profile.get("train_file")
    if not train_file:
        return undefined("no training table is named, so there is no target to read")
    candidates = [
        Path(str(train_file)),
        root / str(train_file),
        root / "data" / "raw" / str(train_file),
    ]
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        return undefined(f"the training table {train_file!r} is not on disk")
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        return undefined(f"could not read {path.name}: {exc}")

    return compute_floor(
        frame,
        target=target,
        plan=plan,
        metric_name=metric_name,
        direction=direction,
        anchor_column=profile.get("anchor_column"),
        num_classes=len((profile.get("target_distribution") or {}).get("class_counts") or {})
        or None,
    )


def write_floor(root: Path, reading: FloorReading) -> Path:
    path = Path(root) / FLOOR_FILENAME
    path.write_text(reading.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_floor(root: Path) -> FloorReading | None:
    path = Path(root) / FLOOR_FILENAME
    if not path.is_file():
        return None
    try:
        return FloorReading.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
