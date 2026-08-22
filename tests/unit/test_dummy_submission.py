"""M24 tier 1's "dummy baseline 100%", read honestly.

The number cannot mean "the floor scored well" — a floor that scored well is a
gate no model can pass. It means the dumbest defensible answer produces a
submission the competition would **accept**: it runs, it has the sample's exact
columns and row count, no NaN, and every label in it was seen in training.

That is a more basic claim than any metric, and a pipeline failing it has a
problem no score would reveal, because there is nothing to measure yet.
"""

from __future__ import annotations

import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.floor import FloorReading
from labpilot.research_engine.execution.baseline.submission import (
    check_submission,
    dummy_submission_is_valid,
    emit_submission,
)


def _floor(strategy: str = "majority_class") -> FloorReading:
    return FloorReading(metric_name="accuracy", score=0.6, best_strategy=strategy)


def _labels(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({"Id": range(n), "y": [0, 1, 1, 1] * (n // 4)})


def _sample(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"Id": range(n), "y": [0] * n})


# --- what it emits -------------------------------------------------------------


def test_the_submission_takes_its_shape_from_the_sample() -> None:
    """The sample is the competition's own statement of the shape it wants.

    Rebuilding it from the test table is how a submission comes to have the right
    values in the wrong shape — and the wrong shape is rejected without ever
    being scored.
    """
    submission = emit_submission(_floor(), _labels()["y"], _sample(), target_column="y")

    assert list(submission.columns) == ["Id", "y"]
    assert len(submission) == 5
    assert list(submission["Id"]) == [0, 1, 2, 3, 4]


def test_the_constant_is_fitted_on_the_whole_training_target() -> None:
    """Unlike the floor, which holds rows out because it is estimating
    generalisation. A submission estimates nothing, and withholding data from it
    would answer with less than is known."""
    submission = emit_submission(_floor(), _labels()["y"], _sample(), target_column="y")

    assert set(submission["y"]) == {1}, "the majority class over all 20 rows"


def test_a_sample_without_the_target_is_refused() -> None:
    sample = pd.DataFrame({"Id": [1, 2]})

    with pytest.raises(ValueError, match="no 'y' column"):
        emit_submission(_floor(), _labels()["y"], sample, target_column="y")


# --- what it refuses -----------------------------------------------------------


def test_a_valid_submission_has_no_reasons() -> None:
    train = _labels()
    check = dummy_submission_is_valid(_floor(), train, _sample(), target_column="y")

    assert check.valid and check.reasons == ()


def test_every_reason_is_reported_not_just_the_first() -> None:
    """An operator fixing a submission wants the list; stopping at the first
    sends them round the loop once per problem."""
    sample = _sample(5)
    wrong = pd.DataFrame({"Id": [1, 2], "y": [None, 99]})

    check = check_submission(wrong, sample, _labels()["y"], target_column="y")

    assert not check.valid
    assert len(check.reasons) >= 3, check.reasons
    assert any("row(s)" in r for r in check.reasons)
    assert any("NaN" in r for r in check.reasons)
    assert any("never seen in training" in r for r in check.reasons)


def test_a_wrong_column_set_is_a_reason() -> None:
    check = check_submission(
        pd.DataFrame({"Id": range(5), "prediction": [1] * 5}),
        _sample(5),
        _labels()["y"],
        target_column="y",
    )

    assert not check.valid
    assert "columns are" in check.reasons[0]


def test_a_label_never_seen_in_training_is_refused() -> None:
    """The check that catches a real class of mistake: a regression constant
    written into a classification target is numerically fine and entirely
    inadmissible, and no shape check would notice."""
    check = check_submission(
        pd.DataFrame({"Id": range(5), "y": [7] * 5}),
        _sample(5),
        _labels()["y"],
        target_column="y",
    )

    assert not check.valid
    assert "never seen in training" in check.reasons[0]


def test_a_continuous_target_is_not_asked_about_labels() -> None:
    """Review of my own first version: `SalePrice` is an **integer** column with
    663 distinct values, so a dtype test called it discrete and rejected the
    floor's own constant as a label never seen.

    That is the `SalePrice` misreading M23 step 1 exists to prevent, one layer
    down — so the rule here is `target_type`'s, and `DISCRETE_LABEL_CEILING` is
    imported rather than restated.
    """
    prices = pd.Series([34900 + 100 * i for i in range(200)])
    submission = pd.DataFrame({"Id": range(5), "y": [166716.7] * 5})

    check = check_submission(submission, _sample(5), prices, target_column="y")

    assert check.valid, check.reasons


# --- what it cannot answer -------------------------------------------------------


def test_a_capture_with_no_rows_says_so_rather_than_failing() -> None:
    """Every fixture in the corpus is headers-only, so this is the answer the
    hermetic tier gets. Scoring it as a miss would be measuring the truncation,
    which is the rule the whole corpus runs on.
    """
    check = dummy_submission_is_valid(
        _floor(), _labels().iloc[:0], _sample().iloc[:0], target_column="y"
    )

    assert not check.valid
    assert "no rows" in check.reasons[0]


def test_a_missing_target_column_is_a_reason_not_a_crash() -> None:
    check = dummy_submission_is_valid(
        _floor(), pd.DataFrame({"Id": [1, 2]}), _sample(), target_column="y"
    )

    assert not check.valid
    assert "no 'y' column" in check.reasons[0]


def test_a_strategy_with_no_constant_is_a_reason_not_a_crash() -> None:
    """`class_prior` predicts probabilities, not a point value — asking it for a
    constant is a question it has no answer to."""
    check = dummy_submission_is_valid(
        _floor("class_prior"), _labels(), _sample(), target_column="y"
    )

    assert not check.valid
    assert "no constant" in check.reasons[0]
