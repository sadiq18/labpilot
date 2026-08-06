"""Resolving a plan's technique into something the template path can render.

The measured failure this closes: twelve distinct hypotheses on rogii scored
**MSE 194.80 identically**, because `_render_template_fallback` received no
plan metadata and the renderer call discarded every keyword argument it
already accepted. The run looked healthy and tested nothing.

Tests assert the *resolution*, never that a kwarg was forwarded — forwarding is
what the previous code appeared to do.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from labpilot.research_engine.execution.technique.resolver import (
    requested_technique,
    resolve_technique,
)

# rogii's real shape: all-numeric feature columns, partitioned, no categoricals.
ROGII_PROFILE = {
    "target_column": "TVT",
    "id_column": "row_id",
    "partitioned": True,
    "columns": [
        {"name": "MD", "dtype": "float64"},
        {"name": "GR", "dtype": "float64"},
        {"name": "TVT", "dtype": "float64", "is_target": True},
    ],
}
CATEGORICAL_PROFILE = {
    "target_column": "y",
    "partitioned": False,
    "columns": [
        {"name": "city", "dtype": "object"},
        {"name": "age", "dtype": "int64"},
        {"name": "y", "dtype": "float64", "is_target": True},
    ],
}


def _choice(**kw):
    kw.setdefault("problem_type", "tabular_regression")
    kw.setdefault("partitioned", True)
    kw.setdefault("validation", SimpleNamespace(exclude_features=[]))
    return SimpleNamespace(**kw)


# --- the four outcomes, and why they must stay distinct ---------------------


def test_no_technique_resolves_to_none_and_changes_nothing():
    """N5: a plan with no technique must render exactly what it renders today."""
    res = resolve_technique({}, {}, choice=_choice(), profile=ROGII_PROFILE)
    assert res.status == "none"
    assert res.changes_rendering is False


def test_applicable_technique_yields_recipes():
    res = resolve_technique(
        {"technique": "lag_features"}, {}, choice=_choice(), profile=ROGII_PROFILE
    )
    assert res.status == "applied"
    assert res.canonical == "lag_features"
    assert res.feature_recipes == ["lag_features"]
    assert res.changes_rendering is True


def test_record_reference_is_rejected_not_treated_as_a_candidate():
    """`hyp:H-010` is provably not a technique — the one case that asserts junk.

    Six of rogii's ten plans carried exactly this.
    """
    res = resolve_technique({"technique": "hyp:H-010"}, {}, choice=_choice())
    assert res.status == "rejected"
    assert res.changes_rendering is False
    assert "record reference" in res.reason


def test_unknown_technique_is_a_candidate_not_a_rejection():
    """The distinction that keeps a new method from a paper out of the junk bin.

    It has no deterministic recipe, so the template path renders unchanged and
    codegen implements it from the description — but it is never called junk.
    """
    res = resolve_technique({"technique": "focal_loss"}, {}, choice=_choice())
    assert res.status == "candidate"
    assert res.changes_rendering is False
    assert "focal_loss" in res.reason


def test_technique_needing_categoricals_is_not_applicable_to_rogii():
    """Applied-with-no-effect must not read as applied.

    rogii is all-numeric, so target encoding cannot do anything. Recording this
    as `not_applicable` is what stops reflection concluding "target encoding
    does not help" from a run where it never ran.
    """
    res = resolve_technique(
        {"technique": "target_encoding"}, {}, choice=_choice(), profile=ROGII_PROFILE
    )
    assert res.status == "not_applicable"
    assert "categorical_columns" in res.reason
    assert res.changes_rendering is False


def test_same_technique_is_applicable_where_the_data_supports_it():
    """Control for the test above — without it, `not_applicable` could be
    returned for any reason at all."""
    res = resolve_technique(
        {"technique": "target_encoding"},
        {},
        choice=_choice(partitioned=False),
        profile=CATEGORICAL_PROFILE,
    )
    assert res.status == "applied"
    assert res.feature_recipes == ["target_encoding"]


def test_partition_technique_rejected_on_unpartitioned_data():
    res = resolve_technique(
        {"technique": "rolling_features"},
        {},
        choice=_choice(partitioned=False),
        profile=CATEGORICAL_PROFILE,
    )
    assert res.status == "not_applicable"
    assert "partitioned" in res.reason


def test_unknown_precondition_fails_closed():
    """A `requires` nobody implemented must block, not silently pass."""
    from labpilot.research_engine.execution.technique.resolver import _precondition_met

    assert _precondition_met("some_future_check", ROGII_PROFILE, _choice()) is False


# --- identity: aliases collapse so evidence can accumulate -------------------


def test_aliases_resolve_to_one_canonical_identity():
    """Three spellings of one method must not become three findings."""
    canon = {
        resolve_technique(
            {"technique": spelling}, {}, choice=_choice(), profile=ROGII_PROFILE
        ).canonical
        for spelling in ("rolling_features", "rolling", "rolling window")
    }
    assert canon == {"rolling_features"}


# --- precedence: mirrors what the LLM prompt already uses -------------------


@pytest.mark.parametrize(
    ("plan_meta", "expected"),
    [
        ({"technique": "lag_features"}, "lag_features"),
        ({"technique_stack": ["vit", "rolling_features"]}, "rolling_features"),
        ({"combo_techniques": ["binning"]}, "binning"),
        ({}, ""),
    ],
)
def test_requested_technique_precedence(plan_meta, expected):
    assert requested_technique(plan_meta, {}) == expected


# --- the template must be able to act on the recipe -------------------------
#
# Removed here: a test asserting F7 by putting a *recipe name* into
# `exclude_features`. Real exclude lists hold column names (TVT, ANCC), so that
# check could never fire on live data — it asserted a safety property that was
# structurally unable to fail. F7 is enforced in the templates instead; see the
# resolver module docstring.


def test_recipe_without_a_template_gate_is_not_reported_as_applied():
    """rogii's exact case, and the reason it matters.

    `lag_features` passes every precondition on the partitioned dataset, is
    handed to the renderer, and the template — which has zero gates — ignores
    it. Reporting `applied` would record "the technique ran and changed
    nothing" as evidence about the technique.
    """
    res = resolve_technique(
        {"technique": "lag_features"},
        {},
        choice=_choice(template_name="tabular_regression_partitioned"),
        profile=ROGII_PROFILE,
    )
    assert res.status == "not_applicable"
    assert "no gate" in res.reason
    assert res.changes_rendering is False


def test_recipe_with_a_template_gate_is_applied():
    """Control: the gate check must not reject everything."""
    res = resolve_technique(
        {"technique": "target_encoding"},
        {},
        choice=_choice(partitioned=False, template_name="tabular_regression"),
        profile=CATEGORICAL_PROFILE,
    )
    assert res.status == "applied", res.reason
    assert res.feature_recipes == ["target_encoding"]


def test_gate_detection_reads_the_real_templates():
    """The gate map is derived from template source, so it cannot drift."""
    from labpilot.research_engine.execution.technique.registry import gated_recipes

    assert gated_recipes("tabular_regression") == {"log_numeric", "target_encoding"}
    assert gated_recipes("tabular_regression_partitioned") == frozenset()
    assert gated_recipes("no_such_template") == frozenset()
