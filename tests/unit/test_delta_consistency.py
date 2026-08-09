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


# --- how an ensemble is spelled ---------------------------------------------

_TWO_MODELS = """
import lightgbm as lgb
import catboost as cb
import numpy as np


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    b = cb.CatBoostRegressor()
    b.fit(X, y)
"""


@pytest.mark.parametrize(
    ("label", "ending", "combined"),
    [
        ("named aggregator", "    return np.mean([a.predict(X), b.predict(X)], axis=0)\n", True),
        ("plain average", "    return (a.predict(X) + b.predict(X)) / 2\n", True),
        ("weighted blend", "    return 0.6 * a.predict(X) + 0.4 * b.predict(X)\n", True),
        ("scalar times sum", "    return 0.5 * (a.predict(X) + b.predict(X))\n", True),
        (
            "via locals",
            "    pa = a.predict(X)\n    pb = b.predict(X)\n    return (pa + pb) / 2\n",
            True,
        ),
        ("second model discarded", "    return a.predict(X)\n", False),
        ("one model, scaled", "    return a.predict(X) * 2\n", False),
    ],
)
def test_every_spelling_of_an_ensemble_is_recognised(label, ending, combined):
    """Checking only for `mean`/`stack` rejected four of five correct forms.

    The weighted blend is the one that matters: it is the standard technique
    and never calls an aggregator. Each false violation costs a re-ask, and
    steps are the scarce resource in a campaign.

    The last case is the guard against over-correcting — `a.predict(X) * 2`
    is arithmetic over two names (`a` and `X`) but combines nothing, so
    counting parameters would let a scaled single model pass as an ensemble.
    """
    report = check_delta_consistency(PARENT, _TWO_MODELS + ending, **CLAIM)
    assert report.ok is combined, report.violations


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


# --- a change that cannot run is not a change --------------------------------

_PARENT = """
import pandas as pd


def engineer_features(df):
    return df


def main():
    df = pd.read_csv("x.csv")
    cols = [c for c in df.columns]
    return cols


if __name__ == "__main__":
    main()
"""

#: The real shape of rogii 2026-08-09: thirty-four correct lines written into a
#: function `main()` never calls.
_DEAD = """
import pandas as pd


def engineer_features(df):
    df = df.copy()
    for col in ["MD", "GR"]:
        df[f"{col}_roll_mean"] = df.groupby("partition_id")[col].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
    return df


def main():
    df = pd.read_csv("x.csv")
    cols = [c for c in df.columns]
    return cols


if __name__ == "__main__":
    main()
"""

_WIRED = _DEAD.replace(
    '    df = pd.read_csv("x.csv")\n',
    '    df = pd.read_csv("x.csv")\n    df = engineer_features(df)\n',
)


def test_a_delta_into_a_function_nothing_calls_is_a_violation():
    """The first delta the adapter produced against a real pipeline. It parsed,
    it applied, and it could not execute."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    report = check_delta_consistency(_PARENT, _DEAD)

    assert report.ok is False
    assert any("cannot execute" in v for v in report.violations)


def test_the_same_delta_passes_once_it_is_wired_in():
    """The carve-out must not cost the behaviour it guards: the identical
    feature code, called from `main`, is a real change."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    report = check_delta_consistency(_PARENT, _WIRED)

    assert report.ok is True, report.violations


def test_a_better_claim_would_not_have_caught_it():
    """Why this cannot live in the claim.

    `check_addition` looks for the claimed symbols in the child. Name the real
    contribution — `rolling`, `groupby` — and both are present in the dead
    function, so every claim-based check passes on code that never runs. The
    accident that caught it was a *bad* claim naming the container.
    """
    import ast

    from labpilot.research_engine.execution.delta.consistency import check_addition

    assert check_addition(ast.parse(_DEAD), ["rolling", "groupby"]) == []


def test_one_dead_helper_among_live_changes_is_not_a_violation():
    """Conservative on purpose. A delta that edits two functions and leaves one
    helper uncalled has still changed behaviour, and a violation there is a
    re-ask spent on a correct experiment."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _WIRED.replace(
        "def main():",
        "def _unused_helper(a):\n    return a * 2\n\n\ndef main():",
    )

    report = check_delta_consistency(_PARENT, child)

    assert report.ok is True, report.violations


def test_a_delta_that_touches_nothing_is_not_accused():
    """No touched functions means nothing to reach — an import-only or
    constant-only change is not this failure."""
    import ast

    from labpilot.research_engine.execution.delta.consistency import check_reachability

    assert check_reachability(ast.parse(_PARENT), []) == []


def test_a_library_module_is_never_accused():
    """A module that runs nothing of its own cannot establish that a function
    is unreachable — its callers are elsewhere. Without this precondition the
    check condemned most of the fixtures in this file."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    parent = "def train(X):\n    return X\n"
    child = "def train(X):\n    return X * 2\n"

    report = check_delta_consistency(parent, child)

    assert report.ok is True, report.violations


def test_a_module_level_call_counts_as_an_entry_point():
    """Not every script uses the __main__ guard."""
    import ast

    from labpilot.research_engine.execution.delta.consistency import _has_entry_point

    assert _has_entry_point(ast.parse("def main():\n    pass\n\n\nmain()\n")) is True
    assert _has_entry_point(ast.parse("def main():\n    pass\n")) is False


# --- a change that cannot alter behaviour is not an experiment ---------------

_EFFECT_PARENT = '''"""Keeps H-014 feature set and adds Decision Tree."""

import pandas as pd


def add_rolling_features(df):
    return df.groupby("pid")["GR"].transform(lambda x: x.rolling(3).mean())


def main():
    df = pd.read_csv("x.csv")
    return add_rolling_features(df)


if __name__ == "__main__":
    main()
'''


def test_a_docstring_only_delta_is_a_violation():
    """H-015's third and final attempt, measured on rogii 2026-08-09.

    Handed the LightGBM dtype error as its retry reason, aider edited the
    module docstring and nothing else. It consumed the hypothesis's last
    attempt, and the hypothesis was retired as having failed three times when
    it had really been tested twice.
    """
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _EFFECT_PARENT.replace("adds Decision Tree", "adds rolling features")

    report = check_delta_consistency(_EFFECT_PARENT, child, add=["rolling", "groupby"])

    assert report.ok is False
    assert any("no executable code" in v for v in report.violations)


def test_the_checks_that_missed_it_still_miss_it():
    """Why it needed its own check rather than a wider existing one.

    Every other verdict is about *how* the code changed. Nothing touched, the
    claimed symbols present from an earlier attempt, and an edit did happen —
    so each of them passes on its own terms, correctly.
    """
    import ast

    from labpilot.research_engine.execution.delta.consistency import (
        check_addition,
        check_reachability,
        touched_functions,
    )

    parent = ast.parse(_EFFECT_PARENT)
    child = ast.parse(_EFFECT_PARENT.replace("adds Decision Tree", "adds rolling features"))

    assert touched_functions(parent, child) == []
    assert check_reachability(child, []) == []
    assert check_addition(child, ["rolling", "groupby"]) == []


def test_a_comment_only_delta_is_a_violation():
    """Comments never reach the AST, so this is the same failure arriving in a
    form the dump comparison cannot see at all."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _EFFECT_PARENT.replace(
        "    df = pd.read_csv", "    # tuned for the new features\n    df = pd.read_csv"
    )

    report = check_delta_consistency(_EFFECT_PARENT, child, add=["rolling"])

    assert report.ok is False


def test_a_reformatted_delta_is_a_violation():
    """Whitespace is not an experiment either."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _EFFECT_PARENT.replace("def main():\n", "def main():\n\n")

    assert check_delta_consistency(_EFFECT_PARENT, child, add=["rolling"]).ok is False


def test_a_one_character_behaviour_change_passes():
    """The carve-out must be exact: the smallest real change is still a change,
    and a check that fired on it would cost a re-ask on a correct experiment."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _EFFECT_PARENT.replace("rolling(3)", "rolling(5)")

    report = check_delta_consistency(_EFFECT_PARENT, child, add=["rolling"])

    assert report.ok is True, report.violations


def test_a_docstring_change_alongside_a_real_change_passes():
    """Deltas routinely update the docstring they are also implementing."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    child = _EFFECT_PARENT.replace("adds Decision Tree", "adds rolling features").replace(
        "rolling(3)", "rolling(5)"
    )

    assert check_delta_consistency(_EFFECT_PARENT, child, add=["rolling"]).ok is True


def test_a_baseline_has_no_parent_to_be_identical_to():
    """With no parent the question does not arise, and asking it would fail
    every baseline."""
    from labpilot.research_engine.execution.delta.consistency import check_delta_consistency

    assert check_delta_consistency("", _EFFECT_PARENT, add=["rolling"]).ok is True
