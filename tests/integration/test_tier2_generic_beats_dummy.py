"""Tier 2 — the truth the hermetic corpus is checked against.

The plan: *"Tier 2 — full data, nightly. Row counts, cardinality, distributions,
real media probing, undecimated rogii, and **generic-beats-dummy**, defined as
strictly better in the metric's declared direction by more than the fold-to-fold
std. Not 'better by any epsilon' — that is noise."*

Tier 1 explicitly does not make this claim: *"on 50 rows LightGBM routinely loses
to the mean, and asserting it there is asserting noise."* So it lives here, on
whole datasets, and it is the row of M24's acceptance table that a headers-only
corpus can never fill.

**Skipped loudly** when a dataset is absent, naming every path it looked in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from labpilot.accessor.benchmark.fixture import CompetitionFixture, load_fixture
from labpilot.research_engine.execution.baseline.baseline_one import (
    beats_floor_beyond_noise,
    fit_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import compute_floor
from labpilot.research_engine.execution.baseline.selector import ValidationPlan

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "competitions"
_ENV_ROOT = "LABPILOT_CORPUS_FULL_DATA"
_CACHE_ENV = "LABPILOT_KAGGLE_CACHE_DIR"
_DEFAULT_CACHE = Path.home() / "workspace" / ".labpilot-cache" / "kaggle"

#: `(slug, target_type)`. The shape a generic model can be fitted on today: one
#: training table, one target column. rogii is partitioned and playground ships a
#: known-failure metric, so both need work this file does not have — named rather
#: than omitted, and `test_every_competition_is_accounted_for` keeps the list
#: honest as the corpus grows.
_FITTABLE = (
    ("titanic", "binary"),
    ("spaceship-titanic", "binary"),
    ("house-prices-advanced-regression-techniques", "continuous"),
)


def _candidates(fixture: CompetitionFixture) -> list[Path]:
    found: list[Path] = []
    override = os.environ.get(_ENV_ROOT)
    if override:
        found.append(Path(override) / fixture.slug)
    found.append(Path(os.environ.get(_CACHE_ENV) or _DEFAULT_CACHE) / fixture.slug)
    if fixture.source and not fixture.source.startswith("kaggle"):
        found.append(Path(fixture.source))
    return found


def _dataset(fixture: CompetitionFixture) -> Path:
    for candidate in _candidates(fixture):
        if candidate.is_dir() and (candidate / "train.csv").is_file():
            return candidate
    pytest.skip(
        f"no full dataset for {fixture.slug!r}; looked in "
        + ", ".join(str(c) for c in _candidates(fixture))
        + f". Set {_ENV_ROOT} to a directory holding <slug>/ per competition."
    )


def _metric(slug: str) -> tuple[str, str]:
    spec = json.loads((CORPUS / slug / "data" / "competition.json").read_text(encoding="utf-8"))
    metric = spec.get("evaluation_metric") or {}
    return str(metric.get("key") or ""), str(metric.get("direction") or "")


def _readings(slug: str, target_type: str):
    fixture = load_fixture(CORPUS / slug)
    dataset = _dataset(fixture)
    target = fixture.expected.target_column
    metric, direction = _metric(slug)
    train = pd.read_csv(dataset / "train.csv")
    excluded = set(fixture.expected.id_columns or []) | {target}
    plan = ValidationPlan(scheme="kfold", n_splits=5)
    classes = 2 if target_type == "binary" else None

    floor = compute_floor(
        train,
        target=target,
        plan=plan,
        metric_name=metric,
        direction=direction,
        num_classes=classes,
    )
    model = fit_baseline_one(
        train,
        target=target,
        plan=plan,
        metric_name=metric,
        target_type=target_type,
        feature_columns=[c for c in train.columns if c not in excluded],
        num_classes=classes,
    )
    return floor, model, direction


@pytest.mark.tier2
@pytest.mark.parametrize(("slug", "target_type"), _FITTABLE)
def test_the_generic_model_beats_the_floor_by_more_than_noise(slug: str, target_type: str) -> None:
    """M24's *"generic baseline: consistently beats dummy"*.

    "Consistently" is the whole difference between this and a comparison of two
    numbers: a model whose folds disagree by more than its margin over the floor
    has not shown it is better, it has shown the folds disagree.
    """
    floor, model, direction = _readings(slug, target_type)

    assert floor.is_defined, floor.undefined_reason
    assert model.is_defined, model.undefined_reason

    margin = beats_floor_beyond_noise(floor, model, direction)

    assert margin.beats_noise, (
        f"{slug}: floor {floor.score:.4f}, model {model.score:.4f}, "
        f"gap {margin.gap:+.4f} against a fold spread of {margin.noise:.4f}"
    )


@pytest.mark.tier2
@pytest.mark.parametrize(("slug", "target_type"), _FITTABLE)
def test_the_generic_model_fits_at_all(slug: str, target_type: str) -> None:
    """The more basic claim, and the one that was silently false.

    `_prepare` tested `column.dtype == object`, which pandas 3.0 answers `False`
    for a string column — it reports those as `str`. So LightGBM was handed raw
    strings and refused to fit **every real competition with a text column**,
    which is most of them, and the step-4 tests never noticed because their
    frames were all numeric.
    """
    _floor, model, _direction = _readings(slug, target_type)

    assert model.is_defined, model.undefined_reason
    assert len(model.fold_scores) == 5, "one score per fold, so a bad fold is visible"


@pytest.mark.tier2
def test_the_floor_carries_its_folds_for_the_comparison() -> None:
    """A mean cannot answer "by more than the fold-to-fold std"."""
    floor, _model, _direction = _readings(*_FITTABLE[0])

    assert len(floor.fold_scores) == 5
    assert floor.score == pytest.approx(sum(floor.fold_scores) / 5, abs=1e-9)


@pytest.mark.tier2
def test_every_competition_is_accounted_for() -> None:
    """A competition added and not listed is one whose generic baseline nobody
    checks, and the silence would read as coverage."""
    partitioned = {"rogii-wellbore-geology-prediction"}
    known_red = {"playground-series-s6e7"}
    in_corpus = {p.name for p in CORPUS.iterdir() if (p / "fixture.json").is_file()}

    unaccounted = in_corpus - {slug for slug, _ in _FITTABLE} - partitioned - known_red

    assert not unaccounted, (
        f"in the corpus and checked by nothing here: {sorted(unaccounted)}. "
        "Add them to _FITTABLE, or name why they need a different fit."
    )
