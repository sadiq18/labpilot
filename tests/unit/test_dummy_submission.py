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

    check = check_submission(wrong, sample, _labels()["y"], target_column="y", expects_labels=True)

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
        expects_labels=True,
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
        expects_labels=True,
    )

    assert not check.valid
    assert "never seen in training" in check.reasons[0]


def test_a_regression_constant_is_not_asked_about_labels() -> None:
    """A fractional constant is the *right* answer for a regression metric, and
    the first version of this check rejected it.

    `SalePrice` — an integer column with 663 distinct values — was called
    discrete by a dtype test, and the floor's own mean came back as "a label
    never seen in training". Raising the cardinality bar only moved the line:
    review found an ordinal target (quality 1-8, scored by RMSE) failing the
    same way, because *it* has few repeating values and a fractional optimum.
    No rule reading the target's values can tell those apart. Only the metric
    can, so `expects_labels` comes from the floor's strategy and nowhere else.
    """
    prices = pd.Series([34900 + 100 * i for i in range(200)])
    submission = pd.DataFrame({"Id": range(5), "y": [166716.7] * 5})

    check = check_submission(
        submission, _sample(5), prices, target_column="y", expects_labels=False
    )

    assert check.valid, check.reasons


def test_an_ordinal_target_scored_by_rmse_keeps_its_fractional_constant() -> None:
    """The case that survived the cardinality ceiling: eight repeating values,
    which any value-shape rule calls labels, and a mean that is none of them."""
    quality = pd.Series([1 + (i % 8) for i in range(300)])
    submission = pd.DataFrame({"Id": range(5), "y": [4.5] * 5})

    check = check_submission(
        submission, _sample(5), quality, target_column="y", expects_labels=False
    )

    assert check.valid, check.reasons


# --- what it cannot answer -------------------------------------------------------


def test_a_capture_with_no_rows_cannot_be_checked_either_way() -> None:
    """Every fixture in the corpus is headers-only, so this is the answer the
    hermetic tier gets — and it is `unverifiable`, not invalid. Scoring it as a
    miss would be measuring the truncation, which is the rule the whole corpus
    runs on.
    """
    check = dummy_submission_is_valid(
        _floor(), _labels().iloc[:0], _sample().iloc[:0], target_column="y"
    )

    assert not check.could_be_checked
    assert "no rows" in check.unverifiable_reason
    assert check.reasons == (), "a reason list here would read as a defect"


def test_a_missing_target_column_is_a_reason_not_a_crash() -> None:
    """Unlike the two above, this one *is* a failure: the table the fixture
    names as holding the target does not hold it."""
    check = dummy_submission_is_valid(
        _floor(), pd.DataFrame({"Id": [1, 2]}), _sample(), target_column="y"
    )

    assert check.could_be_checked
    assert not check.valid
    assert "no 'y' column" in check.reasons[0]


@pytest.mark.parametrize(
    ("strategy", "names"),
    [
        ("class_prior", "probability vector"),
        ("constant_prediction", "analytic floor"),
        ("anchor_carry_forward", "value per row"),
    ],
)
def test_a_floor_that_is_not_a_point_prediction_cannot_be_checked(
    strategy: str, names: str
) -> None:
    """Review of my own first version, and the worse half of it.

    These three said `valid=False`, which the scorer records as `fail` — "the
    baseline could not hand in a submission". For an AUC competition that is a
    false accusation against a working pipeline, and AUC is one of the commonest
    metrics on Kaggle; logloss is another, and `anchor_carry_forward` is rogii's
    own winner. None of them has a constant to write into a column, and none of
    them says anything about whether a submission could be produced.
    """
    check = dummy_submission_is_valid(_floor(strategy), _labels(), _sample(), target_column="y")

    assert not check.could_be_checked
    assert names in check.unverifiable_reason
    assert check.reasons == ()


# --- the taxonomy cannot drift from the strategies ------------------------------


def test_every_floor_strategy_is_classified_exactly_once() -> None:
    """`CONSTANT_STRATEGIES` and `NON_CONSTANT_STRATEGIES` decide whether a floor
    can be written into a submission, and they are a second copy of knowledge
    `_constant_for` already has. A strategy added there and forgotten here would
    reach `emit_submission` and raise `unknown floor strategy`, which the checker
    reports as an invalid submission — the exact false accusation this split
    exists to prevent.
    """
    from labpilot.research_engine.execution.baseline import floor as floor_module

    known = set(floor_module.CONSTANT_STRATEGIES) | set(floor_module.NON_CONSTANT_STRATEGIES)
    named = {name for names in floor_module._STRATEGIES_BY_METRIC.values() for name in names}
    named |= {"anchor_carry_forward", "constant_prediction"}

    assert not named - known, f"unclassified floor strategies: {sorted(named - known)}"
    assert not known - named, f"classified but never used: {sorted(known - named)}"
    assert not set(floor_module.CONSTANT_STRATEGIES) & set(floor_module.NON_CONSTANT_STRATEGIES)


def test_every_constant_strategy_really_yields_one() -> None:
    """The other half: a name in `CONSTANT_STRATEGIES` that `_constant_for`
    cannot answer would be routed to `emit_submission` and fail there.
    """
    from labpilot.research_engine.execution.baseline.floor import (
        CONSTANT_STRATEGIES,
        _constant_for,
    )

    values = pd.Series([1.0, 2.0, 2.0, 3.0, 5.0, 8.0])

    for strategy in CONSTANT_STRATEGIES:
        assert _constant_for(strategy, values) is not None, strategy


def test_label_strategies_are_a_subset_of_constant_ones() -> None:
    """A strategy cannot predict a label it has no constant for."""
    from labpilot.research_engine.execution.baseline.floor import (
        CONSTANT_STRATEGIES,
        LABEL_STRATEGIES,
    )

    assert set(LABEL_STRATEGIES) <= set(CONSTANT_STRATEGIES)
