"""What the dummy baseline actually hands in, and whether it is admissible.

M24 tier 1's *"dummy baseline 100%"*, read honestly. The number cannot mean "the
floor scored well" — a floor that scored well would be a floor no model can beat,
which is a broken gate rather than a good one. It means the dumbest defensible
answer **produces a submission the competition would accept**: it runs, it has
exactly the sample's columns and row count, it holds no NaN, and every label in
it was seen in training.

That is a different claim from the floor's score and a more basic one. A pipeline
whose baseline cannot emit a valid file has a problem no metric will reveal,
because there is nothing to measure yet.

**Fitted on the whole training target, unlike the floor.** Cross-validation holds
rows out because it is estimating generalisation; a submission is not estimating
anything, and withholding data from it would be answering with less than is
known. The two constants differ and both are right for what they are for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from labpilot.research_engine.execution.baseline.floor import FloorReading, _constant_for

logger = logging.getLogger(__name__)

__all__ = [
    "SubmissionCheck",
    "check_submission",
    "dummy_submission_is_valid",
    "emit_submission",
]


@dataclass(frozen=True)
class SubmissionCheck:
    """Whether the emitted file is admissible, and every reason it is not.

    All the reasons, not the first: an operator fixing a submission wants the
    list, and a check that stopped at the first failure would send them round the
    loop once per problem.
    """

    valid: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def emit_submission(
    floor: FloorReading,
    train_target: pd.Series,
    sample: pd.DataFrame,
    *,
    target_column: str,
) -> pd.DataFrame:
    """The sample submission, with the target replaced by the floor's constant.

    Built *from* the sample rather than from the test table, because the sample
    is the competition's own statement of the shape it wants — its columns, its
    order, its row count and its ids. Reconstructing that from the test table is
    how a submission comes to have the right values in the wrong shape.
    """
    if target_column not in sample.columns:
        raise ValueError(
            f"the sample submission has no {target_column!r} column; it has {list(sample.columns)}"
        )
    constant = _constant_for(floor.best_strategy, train_target)
    if constant is None:
        raise ValueError(f"the {floor.best_strategy!r} strategy has no constant for this target")
    submission = sample.copy()
    submission[target_column] = constant
    return submission


def _has_labels(values: pd.Series) -> bool:
    """Whether this target is a label set, by the schema's own rule.

    Not `is_float_dtype`. `SalePrice` is an **integer** column with 663 distinct
    values, so a dtype test called it discrete and then rejected the floor's own
    constant as "a label never seen in training" — the exact `SalePrice`
    misreading M23 step 1 exists to prevent, reappearing one layer down.

    So the rule is `target_type`'s: few enough distinct values to be labels, and
    labels that *repeat*. `DISCRETE_LABEL_CEILING` is imported rather than
    restated, because a second copy of that number is how the two would come to
    disagree about what a label is.
    """
    from labpilot.accessor.profiler.tabular import DISCRETE_LABEL_CEILING

    unique = values.nunique()
    return bool(unique) and unique <= DISCRETE_LABEL_CEILING and unique < len(values)


def check_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    train_target: pd.Series,
    *,
    target_column: str,
) -> SubmissionCheck:
    """Every way this file would be rejected, or nothing.

    The label check is the one that catches a real class of mistake: a regression
    constant written into a classification target is numerically fine and
    entirely inadmissible, and no shape check would notice.
    """
    reasons: list[str] = []

    if list(submission.columns) != list(sample.columns):
        reasons.append(
            f"columns are {list(submission.columns)}; the sample wants {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        reasons.append(f"{len(submission)} row(s); the sample has {len(sample)}")
    if target_column in submission.columns and submission[target_column].isna().any():
        count = int(submission[target_column].isna().sum())
        reasons.append(f"{count} row(s) hold NaN in {target_column!r}")

    values = train_target.dropna()
    known = set(values.unique())
    if _has_labels(values) and target_column in submission.columns:
        predicted = set(submission[target_column].dropna().unique())
        unseen = predicted - known
        if unseen:
            reasons.append(
                f"predicts label(s) never seen in training: {sorted(map(str, unseen))[:3]}"
            )

    return SubmissionCheck(valid=not reasons, reasons=tuple(reasons))


def dummy_submission_is_valid(
    floor: FloorReading,
    train: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    target_column: str,
) -> SubmissionCheck:
    """`emit` then `check`, with a failure to emit reported as a reason.

    An exception here is the most basic failure there is — the baseline could not
    produce a file at all — and it belongs in the same list as a wrong column
    count rather than reaching a caller as a traceback.
    """
    if target_column not in train.columns:
        return SubmissionCheck(False, (f"the training table has no {target_column!r} column",))
    if train.empty or sample.empty:
        return SubmissionCheck(
            False,
            (
                "no rows: a headers-only capture cannot emit a submission, "
                "and cannot say whether one would be valid",
            ),
        )
    try:
        submission = emit_submission(
            floor, train[target_column], sample, target_column=target_column
        )
    except (ValueError, KeyError, TypeError) as exc:
        return SubmissionCheck(False, (f"could not emit a submission: {exc}",))
    return check_submission(submission, sample, train[target_column], target_column=target_column)
