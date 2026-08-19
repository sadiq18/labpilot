"""Direction is measured from the evaluator, not looked up in a table.

A metric → direction table is a Kaggle-shaped abstraction: it works for a
platform with a closed metric set and becomes a brittle taxonomy the moment the
objective space is open. The probe replaces the lookup with an observation, which
is what lets a metric nobody has ever catalogued still get a direction.

The test that carries that claim is
`test_a_metric_no_table_could_know_still_gets_a_direction`.
"""

from __future__ import annotations

import numpy as np
import pytest

from labpilot.research_engine.intelligence.competition.direction_probe import (
    degrade,
    probe_direction,
    probe_metric_direction,
)
from labpilot.research_engine.intelligence.competition.metric_vocabulary import _METRICS

_REGRESSION_KEYS = {"rmse", "mse", "mae", "rmsle"}

#: Strictly positive so a log-domain scorer is defined on it.
_CONTINUOUS = np.random.default_rng(7).uniform(1.0, 100.0, size=64)


def _probe(key: str):
    """Give the convenience wrapper the shape facts `compute_metric` asks for."""
    from labpilot.research_engine.intelligence.competition.metric_vocabulary import (
        requires_probabilities,
    )

    classification = key not in _REGRESSION_KEYS
    return probe_metric_direction(
        key,
        needs_probabilities=requires_probabilities(key),
        num_classes=2 if classification else None,
    )


# --- the scaling claim ------------------------------------------------------


def test_a_metric_no_table_could_know_still_gets_a_direction() -> None:
    """The whole point. A domain metric written last week, in no registry
    anywhere, is still resolvable — because the direction is observed rather
    than declared."""

    def wellbore_misfit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Invented, and deliberately not a standard metric."""
        return float(np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred)) ** 1.5))

    probe = probe_direction(wellbore_misfit, degrade(_CONTINUOUS))

    assert probe.direction == "minimize"
    assert probe.confidence >= 0.99
    assert len(probe.evidence) >= 2


def test_a_maximising_unknown_metric_is_read_the_other_way() -> None:
    """Same, in the opposite direction, so the probe is not just returning
    `minimize` for everything continuous."""

    def agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(-np.mean(np.abs(y_true - y_pred)))

    assert probe_direction(agreement, degrade(_CONTINUOUS)).direction == "maximize"


# --- the cross-check against the declared table -----------------------------


@pytest.mark.parametrize(
    "metric", [m for m in _METRICS if m.scorable], ids=lambda m: m.key
)
def test_measurement_confirms_every_declared_direction(metric) -> None:
    """A table entry is a claim; this is an observation. They must agree, and
    where they cannot both be checked the observation is the one to trust."""
    probe = _probe(metric.key)

    assert probe.error is None
    assert probe.direction == metric.direction
    assert probe.confidence >= 0.99


def test_an_unimplemented_metric_reports_why_rather_than_guessing() -> None:
    """`balanced_accuracy` is named in the registry and has no implementation.
    The probe must say so, not fall back to the declared direction."""
    probe = _probe("balanced_accuracy")

    assert probe.indeterminate
    assert probe.confidence == 0.0
    assert probe.error is not None and "Unsupported metric" in probe.error


# --- declining to answer is an answer ---------------------------------------


def test_a_scorer_that_ignores_its_input_is_indeterminate() -> None:
    """No separation between perfect and degraded means the evaluator said
    nothing about direction. Picking one would be inventing evidence."""
    probe = probe_direction(lambda y_true, y_pred: 1.0, degrade(_CONTINUOUS))

    assert probe.indeterminate
    assert probe.confidence == 0.0
    assert "no probe separated" in (probe.error or "")


def test_probes_that_disagree_are_not_resolved_by_majority() -> None:
    """A metric whose direction depends on which degradation you tried is one
    this cannot answer for. Two of three agreeing is still a guess."""
    seen: list[int] = []

    def inconsistent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        seen.append(1)
        # perfect/degraded alternate, so the sign of the gap flips per pair
        return float(len(seen) % 4)

    probe = probe_direction(inconsistent, degrade(_CONTINUOUS))

    assert probe.indeterminate
    assert "disagreed" in (probe.error or "")


def test_a_raising_scorer_surfaces_the_error() -> None:
    def broken(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise RuntimeError("no scorer here")

    probe = probe_direction(broken, degrade(_CONTINUOUS))

    assert probe.indeterminate
    assert "no scorer here" in (probe.error or "")


# --- a reading that moves between runs is not evidence ----------------------


def test_the_probe_is_deterministic() -> None:
    first = _probe("rmse")
    second = _probe("rmse")

    assert [(r.label, r.perfect, r.degraded) for r in first.readings] == [
        (r.label, r.perfect, r.degraded) for r in second.readings
    ]


def test_the_synthetic_truth_never_goes_negative() -> None:
    """`compute_metric` silently computes RMSE when RMSLE meets a negative value.
    A probe straying below zero would measure a different metric than the one it
    names — which happened on this probe's first run.
    """
    from labpilot.research_engine.intelligence.competition.direction_probe import (
        _synthetic_truth,
        degrade,
    )

    truth = _synthetic_truth(None)
    assert truth.min() > 0
    for pair in degrade(truth):
        assert np.asarray(pair.degraded).min() > 0, pair.label


def test_rmsle_is_really_measured(caplog) -> None:
    """The guard above, asserted through the real scorer: no substitution
    warning means RMSLE scored RMSLE."""
    with caplog.at_level("WARNING"):
        probe = _probe("rmsle")

    assert probe.direction == "minimize"
    assert "computing RMSE instead" not in caplog.text


def test_an_unseparated_reading_says_nothing() -> None:
    """Directly covering the branch `_verdict` depends on. Filtering the verdict
    on `separated` instead left this unreachable, and a mutation returning
    "maximize" here kept all seventeen tests green."""
    from labpilot.research_engine.intelligence.competition.direction_probe import ProbeReading

    assert ProbeReading(label="flat", perfect=1.0, degraded=1.0).direction is None
    assert ProbeReading(label="up", perfect=2.0, degraded=1.0).direction == "maximize"
    assert ProbeReading(label="down", perfect=1.0, degraded=2.0).direction == "minimize"


# --- shapes that are neither regression nor classification ------------------


def test_a_ranking_objective_needs_no_task_label() -> None:
    """Ragged rows — a ranked list per query — which neither a regression nor a
    classification branch could have described.

    This is the test that says the taxonomy is gone: `degrade` never looks inside
    an element, so shuffle, roll and collapse are all defined here without the
    probe knowing what a ranking is.
    """
    truth = [[3, 1, 2], [5, 4], [7, 8, 9, 6], [2, 1], [4, 5, 6]]

    def rank_agreement(actual, predicted) -> float:
        """Fraction of rows whose predicted ranking matches the true one."""
        return float(
            np.mean([list(a) == list(b) for a, b in zip(actual, predicted, strict=False)])
        )

    probe = probe_direction(rank_agreement, degrade(truth))

    assert probe.direction == "maximize"
    assert probe.confidence >= 0.99


def test_a_per_row_mask_objective_works_the_same_way() -> None:
    """Segmentation-shaped truth: an array per row, scored by overlap."""
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, size=(24, 8))

    def mean_iou(actual, predicted) -> float:
        actual, predicted = np.asarray(actual), np.asarray(predicted)
        inter = np.logical_and(actual, predicted).sum(axis=1)
        union = np.logical_or(actual, predicted).sum(axis=1)
        return float(np.mean(np.divide(inter, np.maximum(union, 1))))

    assert probe_direction(mean_iou, degrade(truth)).direction == "maximize"


def test_degradations_are_defined_for_ragged_truth() -> None:
    """`np.asarray` cannot make variable-length rows rectangular; the object
    fallback keeps every degradation working on them."""
    pairs = degrade([[1, 2], [3], [4, 5, 6]])

    assert {p.label for p in pairs} == {"shuffle", "roll", "constant"}
    assert all(len(p.degraded) == 3 for p in pairs)
