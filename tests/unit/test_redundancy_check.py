"""A hypothesis whose change is already present must be caught before it runs.

The rogii case, verbatim from aider's own refusal on 2026-08-09:

    The code currently: 1 Trains a LightGBM model … 3 Averages predictions from
    both models … Since no modifications are required, there are no
    SEARCH/REPLACE blocks to output.

Four campaigns spent every step re-selecting that hypothesis, because nothing
marked it implemented. The check below is `check_addition` asked *before* the
experiment rather than after, and it costs nothing.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.delta.redundancy import check_redundancy

_ENSEMBLE = """
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor

def train(X, y):
    a = lgb.train(X, y)
    dt = DecisionTreeRegressor()
    b = dt.fit(X, y).predict(X)
    return (a + b) / 2
"""

_LGB_ONLY = """
import lightgbm as lgb

def train(X, y):
    return lgb.train(X, y)
"""


def test_the_rogii_case_is_caught():
    """Everything the hypothesis promised to add is already there."""
    verdict = check_redundancy(_ENSEMBLE, ["lgb", "DecisionTreeRegressor"])

    assert verdict.redundant is True
    assert "DecisionTreeRegressor" in verdict.reason


def test_a_genuine_new_symbol_is_not_redundant():
    """The experiment that should run: CatBoost is not in the parent."""
    verdict = check_redundancy(_ENSEMBLE, ["CatBoostRegressor"])

    assert verdict.redundant is False


def test_a_partly_present_claim_is_not_redundant():
    """*All*, not *any* — and this is the whole design.

    "Ensemble LightGBM with CatBoost" claims `added=['CatBoostRegressor']` on a
    parent that already has `lgb`. Judging on any present symbol would retire
    the very experiment the hypothesis exists to run.
    """
    verdict = check_redundancy(_ENSEMBLE, ["lgb", "CatBoostRegressor"])

    assert verdict.redundant is False
    assert verdict.already_present == ["lgb"]


def test_the_verdict_names_the_symbol_that_proves_it():
    """A verdict that retires a hypothesis must be checkable by a reader."""
    verdict = check_redundancy(_LGB_ONLY, ["lgb"])

    assert verdict.redundant is True
    assert "'lgb'" in verdict.reason
    assert verdict.already_present == ["lgb"]


@pytest.mark.parametrize("added", [[], None, ["", "   "]])
def test_an_empty_claim_is_never_redundant(added):
    """`DeltaBriefAgent` soft-fails to an empty brief. Reading that as "already
    done" would retire good hypotheses whenever the brief model was
    unavailable."""
    assert check_redundancy(_ENSEMBLE, added).redundant is False


def test_an_unparseable_parent_yields_no_verdict():
    """Guessing "redundant" from a broken file would retire a hypothesis on the
    strength of a syntax error."""
    assert check_redundancy("def broken(:\n", ["lgb"]).redundant is False


def test_an_empty_parent_yields_no_verdict():
    """A baseline has no parent, and nothing is implemented yet."""
    assert check_redundancy("", ["lgb"]).redundant is False


def test_an_imported_but_uncalled_symbol_still_counts():
    """`check_addition` treats import-or-call as present, and this must agree:
    two answers to "is it there?" would drift."""
    verdict = check_redundancy("import catboost\n\ndef f():\n    pass\n", ["catboost"])

    assert verdict.redundant is True


def test_whitespace_in_a_claim_does_not_defeat_the_match():
    assert check_redundancy(_LGB_ONLY, ["  lgb  "]).redundant is True
