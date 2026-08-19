"""Determine which way is better by *measuring* an evaluator, not by looking it up.

A metric → direction table works for a competition platform and becomes a brittle
taxonomy the moment the objective space is open: a domain metric somebody wrote
last week has no entry and never will.

But direction is not something a table has to *know*. Given an evaluator you can
run, it is something you can *observe*:

    perfect  = score(y_true, y_true)
    degraded = score(y_true, something_worse)
    direction = maximize if perfect > degraded else minimize

**The probe knows nothing about tasks.** It takes a scorer and a set of
(truth, perfect, degraded) triples, and the degradations it ships — shuffle, roll,
collapse-to-constant — are defined on *any* array-like truth: floats, class
labels, ranked lists, boxes, masks, episode returns. Nothing here branches on
regression versus classification, because "which way is better" does not depend
on that. An earlier version took `task: Literal["regression", "classification"]`
and reintroduced exactly the taxonomy this module exists to remove — in the one
function whose purpose was to escape it.

The one thing a caller must supply is truth in the shape the scorer expects. That
is knowledge the caller has and the probe cannot invent — and real truth from the
workspace beats any synthetic sample, because it exercises the scorer on the data
it will actually see.

**Several probes, not one.** A single pair can agree by accident: a metric that
saturates, one bounded oddly, one flat over the degradation used. The probe
requires the informative pairs to be *unanimous* and reports `indeterminate`
rather than resolving by majority — a metric whose direction depends on which
degradation you tried is one this cannot answer for, and saying so is the useful
answer. `indeterminate` is the probe declining to guess, which is the point of
measuring in the first place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Direction = Literal["minimize", "maximize"]
Scorer = Callable[[Any, Any], float]

#: Fixed, so two probes of the same evaluator return the same reading. A probe
#: whose answer moves between runs is not evidence.
_SEED = 20260813

#: How far apart two readings must be before a pair counts as informative. Below
#: this the evaluator did not respond to the degradation, and the pair is
#: discarded rather than read as a direction.
_MIN_SEPARATION = 1e-9


@dataclass(frozen=True)
class ProbePair:
    """One (truth, perfect, degraded) triple to score.

    `perfect` is usually the truth itself. It is a separate field because some
    evaluators want predictions in a different shape than labels — probabilities,
    ranked lists, decoded boxes — and the caller is the only one who knows.
    """

    label: str
    y_true: Any
    perfect: Any
    degraded: Any


@dataclass(frozen=True)
class ProbeReading:
    """What the evaluator said about one pair."""

    label: str
    perfect: float
    degraded: float

    @property
    def separated(self) -> bool:
        return abs(self.perfect - self.degraded) > _MIN_SEPARATION

    @property
    def direction(self) -> Direction | None:
        """What this pair says, or None when it says nothing.

        `_verdict` consults this and only this, so an unseparated reading cannot
        reach a verdict through a second, weaker predicate. Filtering the verdict
        on `separated` instead left this None branch unreachable — a mutation
        returning "maximize" from it kept the whole suite green.
        """
        if not self.separated:
            return None
        return "maximize" if self.perfect > self.degraded else "minimize"


@dataclass(frozen=True)
class DirectionProbe:
    """What measuring an evaluator says about which way is better.

    `confidence` is high when the readings agree because this is an observation
    of the scorer that will actually be used, not a claim about a name. It
    outranks any declared direction, and a disagreement between the two is a
    reason to stop rather than to pick one.
    """

    direction: Direction | None
    confidence: float
    readings: tuple[ProbeReading, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def indeterminate(self) -> bool:
        return self.direction is None


def degrade(y_true: Any, *, rng: np.random.Generator | None = None) -> list[ProbePair]:
    """Shape-agnostic degradations of a truth vector.

    Every one is defined for any sequence — floats, class labels, strings, ranked
    lists, per-row masks — because none of them looks *inside* an element. That is
    what lets one probe serve regression, classification, ranking, detection and
    anything else without knowing which it is holding.

    * **shuffle** — the same values, aligned with the wrong rows
    * **roll** — off by one, which degrades an order-sensitive scorer that a
      permutation might happen to leave unchanged
    * **constant** — every row predicted the same, the floor most metrics have
    """
    generator = rng or np.random.default_rng(_SEED)
    arr = _as_array(y_true)
    if arr.shape[0] < 2:
        return []

    order = generator.permutation(arr.shape[0])
    return [
        ProbePair("shuffle", y_true, y_true, arr[order]),
        ProbePair("roll", y_true, y_true, np.roll(arr, 1, axis=0)),
        ProbePair("constant", y_true, y_true, np.repeat(arr[:1], arr.shape[0], axis=0)),
    ]


def _as_array(y_true: Any) -> np.ndarray:
    """A row-indexable view, falling back to object dtype for ragged truth.

    A ranking task's truth is a list of variable-length lists, which numpy cannot
    make rectangular. Object dtype keeps every degradation above working on it,
    because none of them inspects an element.
    """
    try:
        return np.asarray(y_true)
    except (ValueError, TypeError):
        holder = np.empty(len(y_true), dtype=object)
        holder[:] = list(y_true)
        return holder


def probe_direction(scorer: Scorer, pairs: Sequence[ProbePair]) -> DirectionProbe:
    """Measure whether higher or lower is better for `scorer`.

    `scorer` takes ``(y_true, y_pred)`` and returns a float. A pair it cannot
    answer — raising, or returning a non-finite value — is recorded as evidence
    and skipped, not treated as a verdict about the scorer: a metric undefined
    for a constant prediction still has a direction the other pairs can measure.
    Only when *no* pair survives is that an error.
    """
    if not pairs:
        return DirectionProbe(direction=None, confidence=0.0, error="no probe pairs supplied")

    readings: list[ProbeReading] = []
    refusals: list[str] = []
    for pair in pairs:
        try:
            perfect = float(scorer(pair.y_true, pair.perfect))
            degraded = float(scorer(pair.y_true, pair.degraded))
        except Exception as exc:  # noqa: BLE001 — the caller decides what a broken scorer means
            # One degradation the scorer cannot handle is not a verdict about
            # the scorer. A correlation metric is undefined for a constant
            # prediction, and aborting here discarded the shuffle and roll
            # readings that had already agreed — turning a measurable direction
            # into `indeterminate`, which blocks the campaign.
            refusals.append(f"{pair.label}: scorer raised — {exc}")
            continue
        if not (np.isfinite(perfect) and np.isfinite(degraded)):
            refusals.append(f"{pair.label}: scorer returned a non-finite value")
            continue
        readings.append(ProbeReading(label=pair.label, perfect=perfect, degraded=degraded))

    return _verdict(tuple(readings), refusals=tuple(refusals))


def _verdict(
    readings: tuple[ProbeReading, ...], *, refusals: tuple[str, ...] = ()
) -> DirectionProbe:
    """Unanimity or nothing. See the module docstring for why not a majority.

    `refusals` are pairs the scorer could not answer. They are evidence — a
    metric undefined on a constant prediction says something real about it — but
    they do not vote, and they only become an error when nothing else survived.
    """
    informative = [r for r in readings if r.direction is not None]
    if not informative:
        detail = f": {'; '.join(refusals)}" if refusals else ""
        return DirectionProbe(
            direction=None,
            confidence=0.0,
            readings=readings,
            evidence=refusals,
            error=f"no probe separated a perfect prediction from a degraded one{detail}",
        )

    directions = {r.direction for r in informative}
    evidence = tuple(
        f"{r.label}: perfect={r.perfect:.6g} degraded={r.degraded:.6g} → {r.direction}"
        for r in informative
    ) + refusals
    if len(directions) > 1:
        return DirectionProbe(
            direction=None,
            confidence=0.0,
            readings=readings,
            evidence=evidence,
            error="probes disagreed about which way is better",
        )

    direction = directions.pop()
    # Every informative probe agreed, on an evaluator we can actually run. The
    # remaining doubt is whether these degradations resemble real predictions,
    # which is why this is not 1.0.
    confidence = 0.99 if len(informative) >= 2 else 0.80
    return DirectionProbe(
        direction=direction,
        confidence=confidence,
        readings=readings,
        evidence=evidence,
    )


# --- convenience for the one evaluator this repo ships ----------------------


def probe_metric_direction(
    metric_name: str,
    *,
    y_true: Any | None = None,
    needs_probabilities: bool = False,
    num_classes: int | None = None,
) -> DirectionProbe:
    """Probe a metric `execution/metrics.py` implements.

    A **convenience for one scorer**, not the general entry point. `y_proba` and
    `num_classes` are `compute_metric`'s own parameters and belong to it rather
    than to probing — which is why they appear here and not in `probe_direction`.

    Pass `y_true` from the workspace when there is some; the synthetic fallback
    exists only for when there is not.
    """
    from labpilot.research_engine.execution.metrics import compute_metric

    truth = _synthetic_truth(num_classes) if y_true is None else y_true

    def scorer(actual: Any, predicted: Any) -> float:
        predicted_arr = np.asarray(predicted)
        return compute_metric(
            np.asarray(actual),
            predicted_arr,
            metric_name,
            y_proba=predicted_arr.astype(float) if needs_probabilities else None,
            num_classes=num_classes,
        )

    return probe_direction(scorer, degrade(truth))


def _synthetic_truth(num_classes: int | None) -> np.ndarray:
    """A stand-in truth vector when the workspace has none to offer.

    Continuous values are strictly positive so a log-domain metric is defined on
    them. `compute_metric` silently computes RMSE instead when RMSLE meets a
    negative value, so a probe straying below zero would measure a different
    metric than the one it names — the exact substitution this layer exists to
    make visible, walked into by this probe on its first run.
    """
    rng = np.random.default_rng(_SEED)
    if num_classes:
        return rng.integers(0, num_classes, size=64)
    return rng.uniform(1.0, 100.0, size=64)
