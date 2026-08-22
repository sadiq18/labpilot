"""The `dummy_baseline` criterion is actually computed by the scorer.

Review of my own first version found it was not. `score_fixture` took a `dummy`
argument, `_score_directory` was its only caller, and `_score_directory` never
passed one — so every fixture reported `unverifiable` because nothing had run,
and the pass/fail branches were unreachable. Tier 3, which reads the real dataset
with every row it has, reported `unverifiable` for the same non-reason.

That is the failure this whole corpus is built to prevent, one level up: a
criterion that looks measured and is not. It is also invisible to any test that
only asserts the corpus scores `unverifiable` today, because the wrong wiring and
the right one agree on the truncated fixtures — which is why this file builds a
dataset with rows and reads the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from labpilot.accessor.benchmark.fixture import CompetitionFixture, Expectations
from labpilot.accessor.benchmark.score import score_full_dataset


def _dataset(directory: Path, *, target: pd.Series, metric: str, key: str, direction: str) -> None:
    """A minimal competition on disk: three tables and a spec."""
    directory.mkdir(parents=True, exist_ok=True)
    n = len(target)
    train = pd.DataFrame({"Id": range(n), "x": [i % 7 for i in range(n)], "y": target})
    train.to_csv(directory / "train.csv", index=False)
    test = pd.DataFrame({"Id": range(n, n + 20), "x": [i % 7 for i in range(20)]})
    test.to_csv(directory / "test.csv", index=False)
    sample = pd.DataFrame({"Id": range(n, n + 20), "y": [target.iloc[0]] * 20})
    sample.to_csv(directory / "sample_submission.csv", index=False)
    (directory / "competition.json").write_text(
        json.dumps(
            {
                "title": "synthetic",
                "description": "a competition with rows",
                "evaluation_metric": {"name": metric, "key": key, "direction": direction},
            }
        ),
        encoding="utf-8",
    )


def _fixture() -> CompetitionFixture:
    return CompetitionFixture(
        slug="synthetic",
        captured_at="2026-08-22",
        expected=Expectations(target_column="y", id_columns=["Id"]),
    )


def _verdict(card: object) -> tuple[str, str]:
    result = next(r for r in card.results if r.criterion == "dummy_baseline")  # type: ignore[attr-defined]
    return result.verdict, result.detail


def test_a_dataset_with_rows_is_actually_scored(tmp_path: Path) -> None:
    """The regression guard. If nothing computes the check, this reads
    `unverifiable` with "nothing ran" — which is exactly what shipped.
    """
    data = tmp_path / "data"
    _dataset(
        data,
        target=pd.Series([0, 1, 1, 0] * 50),
        metric="categorization_accuracy",
        key="accuracy",
        direction="maximize",
    )

    verdict, detail = _verdict(score_full_dataset(_fixture(), data))

    assert verdict == "pass", detail
    assert "nothing ran" not in detail


def test_a_regression_target_passes_with_a_fractional_constant(tmp_path: Path) -> None:
    """The ordinal case that review found failing: eight repeating values scored
    by RMSE, whose optimal constant is none of them."""
    data = tmp_path / "data"
    _dataset(
        data,
        target=pd.Series([1 + (i % 8) for i in range(200)]),
        metric="root_mean_squared_error",
        key="rmse",
        direction="minimize",
    )

    verdict, detail = _verdict(score_full_dataset(_fixture(), data))

    assert verdict == "pass", detail


@pytest.mark.parametrize(
    ("metric", "key", "direction"),
    [("area_under_curve", "auc", "maximize"), ("log_loss", "logloss", "minimize")],
)
def test_a_floor_without_a_point_prediction_is_unverifiable_not_failed(
    tmp_path: Path, metric: str, key: str, direction: str
) -> None:
    """AUC's floor is a theorem and logloss's is a probability vector. Neither
    has a constant to write into a column, and neither says anything about
    whether the pipeline can hand in a file — so `fail` would be an accusation
    the run does not support.
    """
    data = tmp_path / "data"
    _dataset(data, target=pd.Series([0, 1, 1, 0] * 50), metric=metric, key=key, direction=direction)

    verdict, detail = _verdict(score_full_dataset(_fixture(), data))

    assert verdict == "unverifiable", detail
    assert "strategy" in detail


def test_an_invalid_submission_is_still_recorded_as_a_failure() -> None:
    """`unverifiable` must not become the answer to everything.

    Asserted at the mapping rather than end to end, deliberately. Reaching a
    `fail` through `score_full_dataset` turns out to be hard: the submission
    takes its shape from the sample so the shape is always right, the profiler
    refuses a sample whose columns disagree with the resolved id and target
    before scoring ever starts, and `majority_class` cannot emit a label it did
    not see. A genuine `fail` therefore means `emit_submission` raised — which is
    what a new floor strategy that nobody taught this module about would do, and
    the case worth keeping a verdict for. Inventing a dataset that produced one
    would have meant asserting against a scenario the profiler makes impossible.
    """
    from labpilot.accessor.benchmark.score import _score_dummy
    from labpilot.research_engine.execution.baseline.submission import SubmissionCheck

    result = _score_dummy(SubmissionCheck(False, ("could not emit a submission: boom",)))

    assert result.verdict == "fail"
    assert "boom" in result.detail


def test_the_two_negative_states_are_not_the_same_verdict() -> None:
    """The distinction the fix turns on, pinned in one place: an invalid file is
    `fail`, an unmade check is `unverifiable`, and neither is `pass`."""
    from labpilot.accessor.benchmark.score import _score_dummy
    from labpilot.research_engine.execution.baseline.submission import SubmissionCheck

    invalid = _score_dummy(SubmissionCheck(False, ("3 row(s) hold NaN in 'y'",)))
    unmade = _score_dummy(SubmissionCheck.unverifiable("no rows"))

    assert (invalid.verdict, unmade.verdict) == ("fail", "unverifiable")
