"""Where "dummy baseline 100%" is actually measured.

The hermetic corpus is headers-only, so `dummy_baseline` scores `unverifiable`
for every fixture — with no rows there is no constant to fit and no sample to
shape, and scoring that as a miss would be measuring the truncation. The claim
therefore has to be checked against real data, which is what the plan calls
tier 2.

`100%` is the bar and it is not a metric: every competition's dumbest defensible
answer must produce a submission that would be **accepted**. A pipeline failing
this has a problem no score would reveal, because there is nothing to score yet.

**Skipped loudly** when a dataset is absent, naming every path it looked in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from labpilot.accessor.benchmark.fixture import CompetitionFixture, load_fixture
from labpilot.research_engine.execution.baseline.floor import compute_floor
from labpilot.research_engine.execution.baseline.selector import ValidationPlan
from labpilot.research_engine.execution.baseline.submission import dummy_submission_is_valid

CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "competitions"
_ENV_ROOT = "LABPILOT_CORPUS_FULL_DATA"
#: The default Kaggle cache, hardcoded rather than read from
#: `LABPILOT_KAGGLE_CACHE_DIR`. That variable *cannot* be read here: the autouse
#: `_no_real_dotenv_in_tests` fixture deletes it from the environment so
#: `Settings` cannot pick up a developer's real cache, so a test reading it
#: always saw `None`. The first version of this file read it anyway and printed
#: it in the skip message, which advertised a knob that silently did nothing —
#: an operator who set it got skips naming the very path they had overridden.
#: `_ENV_ROOT` is not on that list and does work, so it is the knob offered.
_DEFAULT_CACHE = Path.home() / "workspace" / ".labpilot-cache" / "kaggle"

#: Competitions whose dumbest answer is a single column of one value. The
#: partitioned and multi-table shapes need a submission builder this does not
#: have yet, and claiming them here would be claiming coverage the code has not
#: got — they are named so the gap is a list rather than a silence.
_SINGLE_TABLE = ("titanic", "spaceship-titanic", "house-prices-advanced-regression-techniques")


def _candidates(fixture: CompetitionFixture) -> list[Path]:
    found: list[Path] = []
    override = os.environ.get(_ENV_ROOT)
    if override:
        found.append(Path(override) / fixture.slug)
    found.append(_DEFAULT_CACHE / fixture.slug)
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


def _sample_path(dataset: Path) -> Path:
    for name in ("sample_submission.csv", "gender_submission.csv"):
        if (dataset / name).is_file():
            return dataset / name
    pytest.skip(f"{dataset.name} has no submission template to shape a submission from")


def _metric(slug: str) -> tuple[str, str]:
    """The competition's own metric, from the captured spec rather than from here."""
    spec = json.loads((CORPUS / slug / "data" / "competition.json").read_text(encoding="utf-8"))
    metric = spec.get("evaluation_metric") or {}
    return str(metric.get("key") or ""), str(metric.get("direction") or "")


@pytest.mark.slow
@pytest.mark.parametrize("slug", _SINGLE_TABLE)
def test_the_dummy_baseline_emits_an_acceptable_submission(slug: str) -> None:
    """The honest reading of "dummy 100%", one competition at a time.

    Not that the floor scored well — a floor that scored well is a gate no model
    can pass — but that the file it produces would be accepted.
    """
    fixture = load_fixture(CORPUS / slug)
    dataset = _dataset(fixture)
    target = fixture.expected.target_column
    assert target, f"{slug} has no expected target to build a submission around"

    metric, direction = _metric(slug)
    assert metric and direction, f"{slug}'s captured spec names no metric"

    train = pd.read_csv(dataset / "train.csv")
    sample = pd.read_csv(_sample_path(dataset))
    floor = compute_floor(
        train,
        target=target,
        plan=ValidationPlan(scheme="kfold", n_splits=5),
        metric_name=metric,
        direction=direction,
    )
    assert floor.is_defined, floor.undefined_reason

    check = dummy_submission_is_valid(floor, train, sample, target_column=target)

    assert check.valid, f"{slug}: " + "; ".join(check.reasons)


@pytest.mark.slow
def test_every_single_table_competition_in_the_corpus_is_covered() -> None:
    """The list above must not quietly fall behind the corpus.

    A competition added and not listed here is one whose dummy baseline nobody
    checks — and the silence would read as coverage, which is the failure mode
    this file's own `_SINGLE_TABLE` comment exists to avoid.
    """
    partitioned = {"rogii-wellbore-geology-prediction"}
    known_red = {"playground-series-s6e7"}
    in_corpus = {p.name for p in CORPUS.iterdir() if (p / "fixture.json").is_file()}

    unaccounted = in_corpus - set(_SINGLE_TABLE) - partitioned - known_red

    assert not unaccounted, (
        f"these competitions are in the corpus and checked by nothing here: {sorted(unaccounted)}. "
        "Add them to _SINGLE_TABLE, or name why they need a different submission builder."
    )
