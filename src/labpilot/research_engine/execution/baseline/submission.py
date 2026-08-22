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

from labpilot.research_engine.execution.baseline.floor import (
    LABEL_STRATEGIES,
    NON_CONSTANT_STRATEGIES,
    FloorReading,
    _constant_for,
)

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

    **Three states, not two.** `unverifiable_reason` is how "the check could not
    be made" stays distinct from "the check was made and the file is bad" — the
    same distinction the corpus draws between `unverifiable` and `fail`, and for
    the same reason. A headers-only capture and an AUC floor both leave nothing
    to check, and reporting either as invalid would accuse a working pipeline of
    being unable to hand in a file.
    """

    valid: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    unverifiable_reason: str = ""

    @property
    def could_be_checked(self) -> bool:
        return not self.unverifiable_reason

    @classmethod
    def unverifiable(cls, reason: str) -> SubmissionCheck:
        return cls(valid=False, unverifiable_reason=reason)


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


def check_submission(
    submission: pd.DataFrame,
    sample: pd.DataFrame,
    train_target: pd.Series,
    *,
    target_column: str,
    expects_labels: bool,
) -> SubmissionCheck:
    """Every way this file would be rejected, or nothing.

    The label check is the one that catches a real class of mistake: a regression
    constant written into a classification target is numerically fine and
    entirely inadmissible, and no shape check would notice.

    **`expects_labels` is required, and comes from the metric.** The first version
    inferred it from the target's values — few enough distinct values, repeating —
    and that rule is wrong in a way real data exposes immediately. An ordinal
    target scored by RMSE (quality 1–8, a small count) has exactly that shape and
    a *fractional* optimal constant, so the inference rejected the floor's own
    answer as "a label never seen in training". That is the `SalePrice` misreading
    M23 step 1 exists to prevent, and raising the cardinality ceiling only hid it
    for targets wide enough to escape: `SalePrice` passed on 663 distinct values,
    not because the rule was right. Only the metric knows whether a prediction is
    a label, so only the metric may answer.
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

    if expects_labels and target_column in submission.columns:
        known = set(train_target.dropna().unique())
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
    """`emit` then `check` — or say why neither could happen.

    An exception from `emit` is the most basic failure there is: the baseline
    could not produce a file at all. It belongs in the reason list rather than
    reaching a caller as a traceback.

    But two cases are **not** that failure, and calling them invalid would accuse
    a working pipeline:

    * **No rows.** A headers-only capture has no target to fit a constant from and
      no sample to shape. There is nothing to check.
    * **A floor that is not a point prediction.** AUC's floor is the analytic 0.5,
      logloss's is a probability vector, and rogii's winner carries a value per
      row. None of them has a constant to write into a column, and none of them
      says anything about whether the pipeline can hand in a file. Reporting them
      as invalid would mark every AUC and logloss competition in the corpus as a
      baseline that cannot emit a submission, which is false and would be the
      loudest wrong signal here.
    """
    if target_column not in train.columns:
        return SubmissionCheck(False, (f"the training table has no {target_column!r} column",))
    if train.empty or sample.empty:
        return SubmissionCheck.unverifiable(
            "no rows: a headers-only capture cannot emit a submission, "
            "and cannot say whether one would be valid"
        )
    if floor.best_strategy in NON_CONSTANT_STRATEGIES:
        return SubmissionCheck.unverifiable(
            f"the floor's {floor.best_strategy!r} strategy "
            f"{NON_CONSTANT_STRATEGIES[floor.best_strategy]}"
        )
    try:
        submission = emit_submission(
            floor, train[target_column], sample, target_column=target_column
        )
    except (ValueError, KeyError, TypeError) as exc:
        return SubmissionCheck(False, (f"could not emit a submission: {exc}",))
    return check_submission(
        submission,
        sample,
        train[target_column],
        target_column=target_column,
        expects_labels=floor.best_strategy in LABEL_STRATEGIES,
    )
