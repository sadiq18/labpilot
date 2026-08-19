"""Determine which way is better by *measuring* an evaluator, not by looking it up.

A metric → direction table works for a competition platform and becomes a brittle
taxonomy anywhere else: the space of objectives is open, and a domain metric
somebody wrote last week has no entry and never will.

But direction is not something a table has to *know*. Given an evaluator you can
run, it is something you can *observe*:

    perfect  = score(y_true, y_true)
    degraded = score(y_true, something_worse)
    direction = maximize if perfect > degraded else minimize

That works for RMSE, for NDCG, for a custom wellbore-geology score — for anything
executable — and it needs no entry anywhere. It is also the only source of
direction that can contradict a declaration and be believed, because a table
entry is a claim and this is an observation.

**Several probes, not one.** A single pair can agree by accident: a metric that
saturates, one that is bounded oddly, one that happens to be flat over the
degradation used. So the probe runs a set of independent perfect/degraded pairs
and requires them to be *unanimous*. Disagreement is reported as
`indeterminate` rather than resolved by majority — a metric whose direction
depends on which degradation you tried is a metric this cannot answer for, and
saying so is the useful answer.

`indeterminate` is not a failure of the probe. It is the probe declining to
guess, which is the whole point of measuring in the first place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Direction = Literal["minimize", "maximize"]
Scorer = Callable[[np.ndarray, np.ndarray], float]

#: Fixed, so two probes of the same evaluator return the same reading. A probe
#: whose answer moves between runs is not evidence.
_SEED = 20260813

#: How far apart two readings must be before the pair counts as informative.
#: Below this the evaluator did not respond to the degradation and the pair is
#: discarded rather than being read as a direction.
_MIN_SEPARATION = 1e-9


@dataclass(frozen=True)
class ProbeReading:
    """One perfect/degraded pair and what the evaluator said about it."""

    label: str
    perfect: float
    degraded: float

    @property
    def separated(self) -> bool:
        return abs(self.perfect - self.degraded) > _MIN_SEPARATION

    @property
    def direction(self) -> Direction | None:
        if not self.separated:
            return None
        return "maximize" if self.perfect > self.degraded else "minimize"


@dataclass(frozen=True)
class DirectionProbe:
    """What measuring an evaluator says about which way is better.

    `confidence` is deliberately high when the readings agree: this is an
    observation of the scorer that will actually be used, not a claim about a
    name. It outranks any declared direction, and a disagreement between the two
    is a reason to stop rather than to pick one.
    """

    direction: Direction | None
    confidence: float
    readings: tuple[ProbeReading, ...] = field(default_factory=tuple)
    evidence: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def indeterminate(self) -> bool:
        return self.direction is None


_Pair = tuple[str, np.ndarray, np.ndarray, np.ndarray]


def _regression_pairs(rng: np.random.Generator) -> list[_Pair]:
    """(label, y_true, perfect_pred, degraded_pred) for continuous targets.

    Values are strictly positive so a log-domain metric (RMSLE) is defined on
    them; a probe that crashes on the metric it is probing tells you nothing.
    """
    y = rng.uniform(1.0, 100.0, size=64)
    # Degradations stay strictly positive. `compute_metric` silently computes
    # RMSE instead when RMSLE meets a negative value, so a probe that wandered
    # below zero would measure a different metric than the one it names — the
    # exact substitution this whole layer exists to make visible, walked into
    # by the probe itself on its first run.
    noisy = np.clip(y + rng.normal(0.0, 10.0, size=y.size), 0.1, None)
    return [
        ("noise", y, y.copy(), noisy),
        ("shuffle", y, y.copy(), rng.permutation(y)),
        ("constant", y, y.copy(), np.full_like(y, float(y.mean()))),
    ]


def _classification_pairs(rng: np.random.Generator, n_classes: int) -> list[_Pair]:
    y = rng.integers(0, n_classes, size=64)
    flipped = (y + 1) % n_classes
    half = y.copy()
    half[: y.size // 2] = (half[: y.size // 2] + 1) % n_classes
    return [
        ("all_wrong", y, y.copy(), flipped),
        ("half_wrong", y, y.copy(), half),
        ("shuffle", y, y.copy(), rng.permutation(y)),
    ]


def probe_direction(
    scorer: Scorer,
    *,
    task: Literal["regression", "classification"] = "regression",
    n_classes: int = 2,
) -> DirectionProbe:
    """Measure whether higher or lower is better for `scorer`.

    `scorer` takes ``(y_true, y_pred)`` and returns a float. Anything it raises
    on is reported as an error rather than swallowed — an evaluator that cannot
    score a trivial input is a fact worth surfacing, not a reason to fall back to
    a default.
    """
    rng = np.random.default_rng(_SEED)
    pairs = (
        _regression_pairs(rng)
        if task == "regression"
        else _classification_pairs(rng, n_classes)
    )

    readings: list[ProbeReading] = []
    for label, y_true, perfect_pred, degraded_pred in pairs:
        try:
            perfect = float(scorer(y_true, perfect_pred))
            degraded = float(scorer(y_true, degraded_pred))
        except Exception as exc:  # noqa: BLE001 — the caller decides what a broken scorer means
            return DirectionProbe(
                direction=None,
                confidence=0.0,
                readings=tuple(readings),
                error=f"scorer raised on the {label!r} probe: {exc}",
            )
        if not (np.isfinite(perfect) and np.isfinite(degraded)):
            continue
        readings.append(ProbeReading(label=label, perfect=perfect, degraded=degraded))

    return _verdict(tuple(readings))


def _verdict(readings: tuple[ProbeReading, ...]) -> DirectionProbe:
    """Unanimity or nothing. See the module docstring for why not a majority."""
    informative = [r for r in readings if r.separated]
    if not informative:
        return DirectionProbe(
            direction=None,
            confidence=0.0,
            readings=readings,
            error="no probe separated a perfect prediction from a degraded one",
        )

    directions = {r.direction for r in informative}
    evidence = tuple(
        f"{r.label}: perfect={r.perfect:.6g} degraded={r.degraded:.6g} → {r.direction}"
        for r in informative
    )
    if len(directions) > 1:
        return DirectionProbe(
            direction=None,
            confidence=0.0,
            readings=readings,
            evidence=evidence,
            error="probes disagreed about which way is better",
        )

    direction = directions.pop()
    # Every informative probe agreed on an evaluator we can actually run. The
    # remaining doubt is whether the probes resemble real predictions, which is
    # why this is not 1.0.
    confidence = 0.99 if len(informative) >= 2 else 0.80
    return DirectionProbe(
        direction=direction,
        confidence=confidence,
        readings=readings,
        evidence=evidence,
    )


def probe_metric_direction(
    metric_name: str,
    *,
    task: Literal["regression", "classification"] = "regression",
    n_classes: int = 2,
) -> DirectionProbe:
    """Probe a metric `execution/metrics.py` already implements.

    Metrics needing probabilities are handed the predictions as their own
    probabilities, which is what a perfectly confident classifier would produce
    and is enough to move the score between a right and a wrong answer.
    """
    from labpilot.research_engine.execution.metrics import compute_metric

    def scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        proba = y_pred.astype(float) if task == "classification" else None
        return compute_metric(
            y_true,
            y_pred,
            metric_name,
            y_proba=proba,
            num_classes=n_classes if task == "classification" else None,
        )

    return probe_direction(scorer, task=task, n_classes=n_classes)
