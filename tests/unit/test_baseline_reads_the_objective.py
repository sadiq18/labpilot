"""M23 step 2: the baseline choice reads the stage before it.

The design's finding, verbatim: *"`selector.py` and the code-engineering
capability contain no reference to an objective at all."* Two questions the
objective had already answered with evidence were being answered a second time
here — the task, from a cardinality rule this file kept its own copy of, and the
metric, by re-reading the contract the objective had already consulted as one of
six ranked sources.

Two implementations of one question is not redundancy. It is two answers, and
nothing was comparing them.
"""

from __future__ import annotations

import pytest

from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile
from labpilot.research_engine.execution.baseline.selector import BaselineSelector
from labpilot.research_engine.intelligence.competition.models import (
    CompetitionSpec,
    MetricSpec,
    ProblemType,
)
from labpilot.research_engine.intelligence.competition.objective import ObjectiveSpec


def _profile(**fields) -> DatasetProfile:
    base = dict(
        competition="demo",
        row_count=100,
        column_count=3,
        target_column="y",
        id_columns=["Id"],
        train_file="train.csv",
        columns=[
            # With `stats`, because the profiler always writes them for a numeric
            # column. Without them `target_type` is `unknown` — a fixture that
            # omitted them was asserting against a profile this system does not
            # produce.
            ColumnProfile(
                name="y",
                dtype="float64",
                unique_count=90,
                is_numeric=True,
                stats={"min": 0.0, "max": 999.5, "mean": 500.0, "std": 100.0},
            ),
        ],
    )
    base.update(fields)
    return DatasetProfile(**base)


def _competition(**fields) -> CompetitionSpec:
    return CompetitionSpec(slug="demo", **fields)


# --- the task -----------------------------------------------------------------


def test_the_objective_decides_the_task(tmp_path) -> None:
    """A measured target beats a keyword match on the description.

    The profile here describes a target with 90 distinct values over 100 rows —
    which this file's own rule reads as regression. The objective, resolved from
    the same data one stage earlier, says classification. Where they disagree the
    one that looked at the column wins, and this is what "one spine" means.
    """
    choice = BaselineSelector().select(
        _competition(),
        _profile(),
        ObjectiveSpec(task="tabular_classification", metric_name="accuracy"),
    )

    assert choice.problem_type == ProblemType.TABULAR_CLASSIFICATION.value
    assert choice.objective_source == "objective"


def test_without_an_objective_nothing_changes(tmp_path) -> None:
    """Every existing caller still works, and still gets the older answer.

    The argument is optional on purpose: this is a swap of precedence, not a
    removal, and a campaign whose workspace has no `objective.json` yet must
    still get a baseline.
    """
    choice = BaselineSelector().select(_competition(), _profile())

    assert choice.problem_type == ProblemType.TABULAR_REGRESSION.value
    assert choice.objective_source == "derived"
    assert choice.objective_metric is None


def test_a_task_with_no_template_falls_through_rather_than_raising() -> None:
    """`image_regression` is an honest task string with nothing to build for it.

    Returning it would raise inside `get_template`, past a caller that reads any
    exception as *"defer to the LLM"* — so an improvement to the objective would
    have silently disabled rule-based selection instead of improving it.
    """
    choice = BaselineSelector().select(
        _competition(),
        _profile(),
        ObjectiveSpec(task="image_regression", metric_name="rmse"),
    )

    assert choice.problem_type == ProblemType.TABULAR_REGRESSION.value, "the older answer"
    assert choice.template_name


# --- the metric ---------------------------------------------------------------


def test_the_objective_decides_the_metric() -> None:
    """Resolved once, from six ranked sources and a probe — not re-read here."""
    choice = BaselineSelector().select(
        _competition(evaluation_metric=MetricSpec(name="RMSE", key="rmse", direction="minimize")),
        _profile(),
        ObjectiveSpec(task="tabular_regression", metric_name="mae"),
    )

    assert choice.metric_name == "mae", "the objective's answer, not the contract's"
    assert choice.objective_metric == "mae"
    assert choice.metric_substituted_from is None


def test_a_metric_the_pipeline_cannot_compute_is_recorded_not_logged() -> None:
    """The metric-mismatch class, made visible one level up.

    An unsupported metric became a default behind a `logger.info`, so CV
    optimised a proxy and no artifact said so. `playground-series-s6e7` states
    balanced accuracy and every campaign scored plain accuracy — and the corpus
    ships that as a known failure precisely because nothing downstream could see
    it. Now the substitution is a field.
    """
    choice = BaselineSelector().select(
        _competition(),
        _profile(),
        ObjectiveSpec(task="tabular_classification", metric_name="balanced_accuracy"),
    )

    assert choice.metric_substituted_from == "balanced_accuracy"
    assert choice.metric_name != "balanced_accuracy", "CV cannot compute it"
    assert choice.objective_metric == "balanced_accuracy", "and the real one is still named"


def test_the_shape_is_read_from_the_profile_not_a_second_threshold() -> None:
    """Review finding. This file kept its own cardinality rule, set to 20.

    The profiler draws the line at 30, so a target with 25 labels was
    classification when `objective.json` existed and regression when it did not —
    two implementations of one question, which is the thing step 2 removes.
    `target_type` is derived, so it is on every profile including ones written
    before it existed; there is nothing to keep a local copy *for*.
    """
    profile = _profile(
        columns=[
            ColumnProfile(
                name="y",
                dtype="int64",
                unique_count=25,
                is_numeric=True,
                stats={"min": 0.0, "max": 24.0},
            )
        ],
        submission_columns=["Id", "y"],
    )

    assert profile.target_type == "multiclass"
    choice = BaselineSelector().select(_competition(), profile)

    assert choice.problem_type == ProblemType.TABULAR_CLASSIFICATION.value


def test_an_unreadable_shape_defers_rather_than_guessing_from_cardinality() -> None:
    """`unknown` means the target's shape could not be read at all.

    Deferring to metadata and modality is the honest next step; applying a
    cardinality rule to a column whose own statistics are unreadable would be
    answering confidently from the one number that survived.
    """
    profile = _profile(
        columns=[ColumnProfile(name="y", dtype="int64", unique_count=90, is_numeric=True)]
    )

    assert profile.target_type == "unknown", "no stats, so no shape"
    assert BaselineSelector().select(_competition(), profile).problem_type


def test_a_supported_metric_records_no_substitution() -> None:
    """The field means "a proxy is being optimised". It must not cry wolf."""
    choice = BaselineSelector().select(
        _competition(),
        _profile(),
        ObjectiveSpec(task="tabular_classification", metric_name="accuracy"),
    )

    assert choice.metric_substituted_from is None


# --- the production path --------------------------------------------------------


def test_an_unresolved_objective_does_not_claim_credit() -> None:
    """Review finding. `objective_source` was set from `objective is not None`.

    A workspace that states no metric resolves to an objective with no task and
    no metric — it blocks launch. Recording `objective` for it said the objective
    drove a decision that came entirely from the older derivation, which is the
    wrong answer to the one question the field exists to answer.
    """
    from labpilot.research_engine.intelligence.competition.objective import resolve_objective

    unresolved = resolve_objective(metric_raw=None)
    assert unresolved.task is None and unresolved.metric_name is None

    choice = BaselineSelector().select(_competition(), _profile(), unresolved)

    assert choice.objective_source == "derived"


def test_the_capability_hands_the_objective_over() -> None:
    """The wiring, checked at the call site rather than assumed.

    `select` taking an objective it is never given would be the same defect one
    layer along: a parameter nothing passes is a parameter nothing uses.
    """
    import inspect

    from labpilot.research_engine.execution.capabilities.code_engineering import capability

    source = inspect.getsource(capability)

    assert "BaselineSelector().select(competition, profile, objective)" in source
    # `ensure_objective`, not `load_objective`. Review finding: the file on disk
    # can be stale by its own recorded inputs, and this was the one consumer
    # that read it without asking.
    assert "ensure_objective(root, context.competition)" in source
    assert "load_objective(root)" not in source


@pytest.mark.parametrize(
    "field", ["objective_source", "objective_metric", "metric_substituted_from"]
)
def test_what_was_read_survives_serialization(field: str) -> None:
    """`baseline_choice.json` is handed whole to the code engineer and read by
    the delta checks. A field that does not serialize reaches neither."""
    import json

    choice = BaselineSelector().select(
        _competition(),
        _profile(),
        ObjectiveSpec(task="tabular_classification", metric_name="balanced_accuracy"),
    )

    assert field in json.loads(choice.model_dump_json())
