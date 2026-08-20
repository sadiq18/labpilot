"""A competent default, and the comparison that is the gate's actual output.

M23 step 4. The floor says a number is not worse than nothing. **Baseline 1 says
whether the pipeline is worth having at all**, and the gate reports the gap:

    Dummy baseline (mean)      RMSE  1.42
    Generic ML     (lightgbm)  RMSE  0.91
    Improvement                35.9%   ✓

Four rules, and three of them are the same ones the floor lives by:

* **LightGBM, already a dependency** — no new install, no GPU, no tuning.
  CatBoost is a later option behind this interface, never a second code path.
* **Minimal preprocessing**, over the `feature_columns` M22 already resolved.
  The point is a competent default, not a good model: anything clever here makes
  the reference move, and a moving reference measures nothing.
* **The same `ValidationPlan`** as the floor and as the pipeline. Three numbers
  on three splits compare nothing.
* **Written here, not by codegen** — putting the reference under the control of
  the thing it measures is how a baseline stops being one.

**Affordability is derived, not assumed.** The plan's own trap: a gate demanding
something unaffordable gets switched off. So where Baseline 1 genuinely cannot
run — an image dataset, or a table past the budget — that is recorded as a
reason, and the gate's `awaiting_ml` state reads it. `awaiting_ml` is not
`passed`. The floor stays the hard requirement because it is cheap and
universal; Baseline 1 is a hard requirement wherever it can run, which on
tabular Kaggle is everywhere.

Plan: ``docs/research-os/autonomy-roadmap/design/18-baseline-correctness.md`` §7.7
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from labpilot.research_engine.execution.baseline.floor import (
    FloorReading,
    fingerprint_of,
    folds_for,
)
from labpilot.research_engine.execution.baseline.selector import ValidationPlan
from labpilot.research_engine.execution.metrics import compute_metric

logger = logging.getLogger(__name__)

__all__ = [
    "BASELINE_ONE_FILENAME",
    "BaselineComparison",
    "ModelReading",
    "affordability",
    "compare",
    "fit_baseline_one",
    "load_baseline_one",
    "write_baseline_one",
]

BASELINE_ONE_FILENAME = "baseline_one.json"

#: Cells (rows × features) above which fitting five folds of LightGBM stops
#: being something to do on the way to a campaign. Deliberately generous — this
#: is a budget, not a benchmark, and the failure mode it guards against is a gate
#: nobody can afford to leave on rather than a slow run.
MAX_AFFORDABLE_CELLS = 20_000_000

#: Modalities whose features are not columns. An image competition's label is
#: still a class column — which is why the *floor* is defined there — but there
#: is nothing for a gradient-boosted tree to read without an extractor, and
#: building one here would be the "anything clever" this module refuses.
_TABULAR_MODALITIES = frozenset({"tabular", ""})

#: Insurance, not the mechanism. At these parameters LightGBM samples nothing —
#: `subsample` and `colsample_bytree` are both 1.0 — so today's determinism comes
#: from the defaults, and removing this seed changes no number. It is set because
#: the day someone adds bagging to "improve" the reference is the day the floor
#: starts moving between runs, and a moving reference measures nothing.
_SEED = 1729


class ModelReading(BaseModel):
    """What a competent default scores. A dataset reading, like the floor.

    Deliberately the same shape as `FloorReading` in the ways that matter — no
    `hypothesis_id`, no execution id, no generated file — because this is the
    control's counterpart and not an experiment result.
    """

    metric_name: str
    model: str = "lightgbm"
    score: float | None = None
    #: Per fold, so a single catastrophic fold is visible rather than averaged
    #: into something that merely looks mediocre.
    fold_scores: list[float] = Field(default_factory=list)
    validation: ValidationPlan = Field(default_factory=ValidationPlan)
    feature_columns: list[str] = Field(default_factory=list)
    fingerprint: str = ""
    computed_at: str = ""
    #: The gate's fingerprint at the moment this was taken — validation
    #: scheme, target, metric, `profile.schema_version` and the M22 answers
    #: fingerprint. Empty means "not recorded", which the gate reports as
    #: unknowable rather than as current: an answer changing the target must
    #: invalidate a reading of the old one, and only this can say that it did.
    workspace_fingerprint: str = ""
    #: Why there is no reading. `awaiting_ml` in the gate reads this — an
    #: unaffordable dataset is a different situation from a failed fit, and both
    #: are different from never having tried.
    undefined_reason: str = ""

    @property
    def is_defined(self) -> bool:
        return self.score is not None and not self.undefined_reason


class BaselineComparison(BaseModel):
    """The gate's output: the floor, the model, and the gap between them.

    **Derived on read, never stored.** A stored verdict is derived state that
    outlives its cause — AGENTS.md rule 2, and the mistake `apply_card_to_beliefs`
    cost this repo. Recompute it from the two readings, which are measurements.
    """

    # Defaulted so an empty comparison is constructible: "nothing has been
    # compared yet" is a real state, and the gate carries one on every verdict
    # including the eight that never reach a comparison at all.
    metric_name: str = ""
    direction: str = ""
    floor_score: float | None = None
    floor_strategy: str = ""
    model_score: float | None = None
    #: Signed so that positive always means *better than the floor*, whichever
    #: way the metric runs. A raw difference would flip meaning with the metric
    #: and every reader would have to remember which one this is.
    improvement: float | None = None
    beats_floor: bool = False
    #: Why no comparison could be made. Empty when both readings are defined.
    incomparable_reason: str = ""

    def render(self) -> str:
        """The three lines from the design, for an operator."""
        if self.incomparable_reason:
            return f"no comparison: {self.incomparable_reason}"
        mark = "OK" if self.beats_floor else "FAIL"
        # A floor of exactly zero leaves the *verdict* standing and only the
        # percentage undefined, so the line has to render without it rather than
        # taking the whole report down with a TypeError.
        gain = "n/a" if self.improvement is None else f"{self.improvement:.1%}"
        return (
            f"Dummy baseline ({self.floor_strategy})  "
            f"{self.metric_name.upper()}  {self.floor_score:.4f}\n"
            f"Generic ML     (lightgbm)  {self.metric_name.upper()}  {self.model_score:.4f}\n"
            f"Improvement                {gain}   {mark}"
        )


def affordability(
    frame: pd.DataFrame, feature_columns: list[str], modality: str = "tabular"
) -> tuple[bool, str]:
    """Whether Baseline 1 can run here, and why not when it cannot.

    Computed rather than assumed, because the plan's trap is real: a gate that
    demands something unaffordable is a gate that gets switched off, and then it
    protects nothing at all.
    """
    if modality not in _TABULAR_MODALITIES:
        return False, (
            f"{modality!r} features are not columns; a tree has nothing to read "
            "without an extractor, and building one here would make the reference move"
        )
    if not feature_columns:
        return False, "no usable feature columns, so there is nothing to fit on"
    cells = len(frame) * len(feature_columns)
    if cells > MAX_AFFORDABLE_CELLS:
        return False, (
            f"{len(frame):,} rows x {len(feature_columns)} features exceeds the "
            f"{MAX_AFFORDABLE_CELLS:,}-cell budget for five folds"
        )
    return True, ""


def _prepare(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Minimal preprocessing: enough for LightGBM to read, and nothing more.

    Object columns become pandas categoricals, which LightGBM handles natively.
    No imputation — it treats NaN as a value, and filling it here would be a
    modelling decision this module has no business making.
    """
    features = frame[feature_columns].copy()
    for name in features.columns:
        column = features[name]
        if column.dtype == object or isinstance(column.dtype, pd.CategoricalDtype):
            features[name] = column.astype("category")
    return features


def _model_fingerprint(
    frame: pd.DataFrame, target: str, features: list[str], plan: ValidationPlan, metric: str
) -> str:
    """The floor's digest, plus the feature set this reading was fitted over.

    `fingerprint_of` rather than a second copy: the previous version repeated
    `to_numpy().tobytes()`, which hashes object *addresses*, so fixing the floor
    alone would have left the two readings disagreeing about whether they
    described the same data. The feature list is the only thing this reading
    depends on that the floor does not.
    """
    digest = hashlib.sha256()
    digest.update(fingerprint_of(frame[target], plan, metric).encode("utf-8"))
    digest.update(",".join(features).encode("utf-8"))
    return digest.hexdigest()


def fit_baseline_one(
    frame: pd.DataFrame,
    *,
    target: str,
    plan: ValidationPlan,
    metric_name: str,
    target_type: str,
    feature_columns: list[str],
    modality: str = "tabular",
    num_classes: int | None = None,
) -> ModelReading:
    """Fit LightGBM per fold on `plan`, and score with `compute_metric`.

    `target_type` decides regressor or classifier — the measured shape from M23
    step 1, not a re-derivation. This is the third consumer of that field, and
    the point of deriving it once was that there would never be a fourth rule.
    """
    now = datetime.now(UTC).isoformat()
    usable = [c for c in feature_columns if c in frame.columns and c != target]
    reading = ModelReading(
        metric_name=metric_name, validation=plan, feature_columns=usable, computed_at=now
    )

    affordable, reason = affordability(frame, usable, modality)
    if not affordable:
        reading.undefined_reason = reason
        return reading
    if target not in frame.columns:
        reading.undefined_reason = f"the training table has no {target!r} column"
        return reading

    classification = target_type in ("binary", "multiclass", "multilabel")
    if target_type in ("none", "unknown", "ordinal", ""):
        reading.undefined_reason = (
            f"target_type is {target_type!r}, so there is no objective to fit toward"
        )
        return reading

    folds = folds_for(plan, frame)
    if not folds:
        reading.undefined_reason = (
            f"the {plan.scheme!r} plan could not be honoured on this table "
            f"({len(frame)} rows, group key {plan.group_key!r})"
        )
        return reading

    reading.fingerprint = _model_fingerprint(frame, target, usable, plan, metric_name)
    features = _prepare(frame, usable)
    y = frame[target]

    import lightgbm as lgb

    common = dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=_SEED,
        n_jobs=1,
        verbose=-1,
    )
    scores: list[float] = []
    for train_idx, val_idx in folds:
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if y_train.isna().all() or y_val.isna().all():
            reading.undefined_reason = "a fold had no usable target values"
            return reading
        if classification and y_train.nunique(dropna=True) < 2:
            reading.undefined_reason = "a training fold contained a single class"
            return reading
        model = (lgb.LGBMClassifier if classification else lgb.LGBMRegressor)(**common)
        try:
            model.fit(features.iloc[train_idx], y_train)
            proba = None
            if classification:
                predicted = model.predict(features.iloc[val_idx])
                probabilities = model.predict_proba(features.iloc[val_idx])
                if probabilities.shape[1] == 2:
                    proba = probabilities[:, 1]
            else:
                predicted = model.predict(features.iloc[val_idx])
            scores.append(
                compute_metric(
                    y_val.to_numpy(),
                    np.asarray(predicted),
                    metric_name,
                    y_proba=proba,
                    num_classes=num_classes,
                )
            )
        except (ValueError, TypeError) as exc:
            # A fit that cannot run is a *reason*, not a traceback: the gate has
            # a state for "no model reading" and none for "the gate crashed".
            logger.info("Baseline 1 could not fit or score a fold: %s", exc)
            reading.undefined_reason = f"lightgbm could not fit or score this dataset: {exc}"
            return reading

    reading.fold_scores = [float(s) for s in scores]
    reading.score = float(np.mean(scores))
    return reading


def compare(
    floor: FloorReading | None, model: ModelReading | None, direction: str
) -> BaselineComparison:
    """The gate's output. Derived on read, never stored.

    `improvement` is signed toward *better*, whichever way the metric runs, so a
    positive number always means the same thing. A raw difference would flip
    meaning with the metric and every reader would have to remember which one
    this is — which is the class of mistake that recorded rogii's only genuine
    improvement as `rejected`.
    """
    comparison = BaselineComparison(
        metric_name=(model.metric_name if model else "") or (floor.metric_name if floor else ""),
        direction=direction,
    )
    if floor is None or not floor.is_defined:
        comparison.incomparable_reason = (
            floor.undefined_reason if floor else "no floor reading"
        ) or "no floor reading"
        return comparison
    comparison.floor_score = floor.score
    comparison.floor_strategy = floor.best_strategy
    if model is None or not model.is_defined:
        comparison.incomparable_reason = (
            model.undefined_reason if model else "no model reading"
        ) or "no model reading"
        return comparison
    if direction not in ("maximize", "minimize"):
        comparison.incomparable_reason = f"direction {direction!r} is not maximize or minimize"
        return comparison
    if floor.metric_name != model.metric_name:
        # Two numbers in different units. The comparison would be arithmetic that
        # means nothing, and it is exactly the mismatch step 2 made visible.
        comparison.incomparable_reason = (
            f"the floor is {floor.metric_name!r} and the model is {model.metric_name!r}"
        )
        return comparison

    comparison.model_score = model.score
    floor_score, model_score = float(floor.score), float(model.score)
    comparison.beats_floor = (
        model_score > floor_score if direction == "maximize" else model_score < floor_score
    )
    if floor_score == 0:
        # A relative improvement over zero is undefined, not infinite. The
        # verdict still stands; only the percentage does not.
        comparison.improvement = None
        return comparison
    gain = (model_score - floor_score) / abs(floor_score)
    comparison.improvement = gain if direction == "maximize" else -gain
    return comparison


def write_baseline_one(root: Path, reading: ModelReading) -> Path:
    path = Path(root) / BASELINE_ONE_FILENAME
    path.write_text(reading.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_baseline_one(root: Path) -> ModelReading | None:
    path = Path(root) / BASELINE_ONE_FILENAME
    if not path.is_file():
        return None
    try:
        return ModelReading.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
