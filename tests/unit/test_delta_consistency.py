"""A delta that applies cleanly is not a delta that tested the hypothesis.

The worked case throughout is *"ensemble LightGBM with CatBoost"*, which has
three plausible bad outcomes that all run, score, and write an evidence card.
The third — added *and* retuned — is the dangerous one: `technique_attribution`
credits the whole `cv_gain` to one technique, so a delta that did more than it
claimed makes that credit false while the number stays real.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.delta.consistency import (
    ConsistencyReport,
    check_delta_consistency,
    imported_modules,
    touched_functions,
)

PARENT = """
import lightgbm as lgb
import numpy as np


def build_features(frame):
    return frame


def train(X, y):
    model = lgb.LGBMRegressor()
    model.fit(X, y)
    return model.predict(X)
"""

SUBSTITUTED = """
import catboost as cb
import numpy as np


def build_features(frame):
    return frame


def train(X, y):
    model = cb.CatBoostRegressor()
    model.fit(X, y)
    return model.predict(X)
"""

ADDED_BUT_UNUSED = """
import lightgbm as lgb
import catboost as cb
import numpy as np


def build_features(frame):
    return frame


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    b = cb.CatBoostRegressor()
    b.fit(X, y)
    return a.predict(X)
"""

ENSEMBLED = """
import lightgbm as lgb
import catboost as cb
import numpy as np


def build_features(frame):
    return frame


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    b = cb.CatBoostRegressor()
    b.fit(X, y)
    return np.mean([a.predict(X), b.predict(X)], axis=0)
"""

CLAIM = {"keep": ["lgb"], "add": ["cb"], "combine": ["lgb", "cb"]}


# --- the three failure modes ------------------------------------------------


def test_substitution_is_caught():
    """Replacing LightGBM measures *substitution* while the card says
    "ensembling improved MSE"."""
    report = check_delta_consistency(PARENT, SUBSTITUTED, **CLAIM)
    assert not report.ok
    assert any("should have been kept" in v for v in report.violations)


def test_added_but_unused_is_caught():
    """The quietest failure: the constructor is present so addition passes, but
    the predictions are discarded and the score reflects the parent alone."""
    report = check_delta_consistency(PARENT, ADDED_BUT_UNUSED, **CLAIM)
    assert not report.ok
    assert any("no aggregation" in v for v in report.violations)


def test_a_correct_ensemble_passes():
    """The test that matters most — a check that rejects everything is not a
    check, it is a blocker."""
    report = check_delta_consistency(PARENT, ENSEMBLED, **CLAIM)
    assert report.ok, report.violations
    assert report.violations == []


def test_a_no_op_delta_claiming_a_technique_is_caught():
    report = check_delta_consistency(PARENT, PARENT, add=["cb"])
    assert not report.ok
    assert any("supposed to be added" in v for v in report.violations)


# --- the bug this file found while being written ----------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import lightgbm as lgb", {"lightgbm", "lgb"}),
        ("import catboost", {"catboost"}),
        ("from lightgbm import LGBMRegressor", {"lightgbm", "LGBMRegressor"}),
        (
            "from sklearn.metrics import mean_squared_error as mse",
            {"sklearn", "sklearn.metrics", "mean_squared_error", "mse"},
        ),
    ],
)
def test_import_aliases_are_collected(source, expected):
    """A hypothesis names a library the way a person would — "keep LightGBM" —
    while the code says `import lightgbm as lgb` and only ever writes `lgb`.

    The first version collected the module name alone, so *every* check failed
    on correct code, including the ensemble that was right.
    """
    import ast

    assert expected <= imported_modules(ast.parse(source))


# --- confinement flags, it does not block -----------------------------------


def test_a_wide_delta_is_flagged_not_refused():
    """A second uncredited change makes attribution false — but a legitimate
    refactor also touches many functions, so blocking would reject real work."""
    wide = (
        ENSEMBLED + "\n" + "\n".join(f"def extra_{i}(x):\n    return x + {i}\n" for i in range(8))
    )
    report = check_delta_consistency(PARENT, wide, **CLAIM)
    assert report.ok, "a wide delta must still be allowed through"
    assert any("touches" in f for f in report.flags)


def test_a_narrow_delta_is_not_flagged():
    report = check_delta_consistency(PARENT, ENSEMBLED, **CLAIM)
    assert report.flags == []


def test_touched_functions_ignores_formatting():
    """Compared on the AST, so a reformat or a comment is not a behaviour
    change — otherwise every delta would look wide."""
    import ast

    reformatted = PARENT.replace("def train(X, y):", "def train(X,  y):  # noqa")
    assert touched_functions(ast.parse(PARENT), ast.parse(reformatted)) == []


def test_touched_functions_names_what_changed():
    import ast

    assert touched_functions(ast.parse(PARENT), ast.parse(ENSEMBLED)) == ["train"]


# --- the report must not fabricate a verdict --------------------------------


def test_an_unparseable_result_is_a_violation():
    report = check_delta_consistency(PARENT, "def broken(:\n", add=["cb"])
    assert not report.ok
    assert any("does not parse" in v for v in report.violations)


def test_a_hypothesis_with_no_claim_yields_no_verdict():
    """Silence is not a pass mark for the delta; it is the absence of anything
    checkable. Inventing a verdict here would be worse than having none."""
    report = check_delta_consistency(PARENT, SUBSTITUTED)
    assert report.ok
    assert report.violations == []


def test_a_missing_parent_still_checks_the_claim():
    """A baseline has no parent, so confinement cannot be computed — but
    addition still can."""
    report = check_delta_consistency("", ENSEMBLED, add=["cb"])
    assert report.ok
    assert report.touched_functions == []


def test_the_report_serialises_for_the_evidence_card():
    report = check_delta_consistency(PARENT, ADDED_BUT_UNUSED, **CLAIM)
    meta = report.as_metadata()
    assert meta["consistent"] is False
    assert meta["violations"] and isinstance(meta["violations"], list)
    assert "train" in meta["touched_functions"]


def test_an_empty_report_is_consistent_by_default():
    assert ConsistencyReport().ok is True
