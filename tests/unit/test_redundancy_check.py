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


# --- dead code implements nothing --------------------------------------------

_DEAD_PARENT = """
import pandas as pd


def engineer_features(df):
    for col in ["MD", "GR"]:
        df[f"{col}_roll"] = df.groupby("pid")[col].transform(
            lambda x: x.rolling(5).mean()
        )
    return df


def main():
    df = pd.read_csv("x.csv")
    return df


if __name__ == "__main__":
    main()
"""


def test_a_symbol_that_only_appears_in_dead_code_is_not_implemented():
    """Measured on rogii 2026-08-09, on the second attempt at H-015.

    The first attempt wrote rolling-window features into `engineer_features`,
    which `main()` never calls; the smoke test failed for an unrelated reason
    and the edit stayed in the workspace. On the retry the brief correctly
    claimed `['rolling', 'groupby']`, both appeared — inside the dead function —
    and the hypothesis was retired as already implemented. The feature it asked
    for had never once been computed.
    """
    from labpilot.research_engine.execution.delta.redundancy import check_redundancy

    verdict = check_redundancy(_DEAD_PARENT, ["rolling", "groupby"])

    assert verdict.redundant is False


def test_the_same_symbols_in_live_code_are_implemented():
    """The carve-out must not cost the behaviour it guards: a parent that really
    does compute rolling features still retires the hypothesis."""
    from labpilot.research_engine.execution.delta.redundancy import check_redundancy

    live = _DEAD_PARENT.replace(
        '    df = pd.read_csv("x.csv")\n',
        '    df = pd.read_csv("x.csv")\n    df = engineer_features(df)\n',
    )

    verdict = check_redundancy(live, ["rolling", "groupby"])

    assert verdict.redundant is True


def test_a_library_parent_is_judged_whole():
    """With no entry point, "nothing calls it here" says only that the caller is
    elsewhere — so nothing is treated as dead."""
    from labpilot.research_engine.execution.delta.redundancy import check_redundancy

    parent = "import lightgbm as lgb\n\n\ndef train(X):\n    return lgb.train(X)\n"

    assert check_redundancy(parent, ["lgb"]).redundant is True


# --- the dead-code question, one level deeper and on the import side ---------

_ENTRY = '\n\ndef main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n'


def test_an_import_used_only_by_dead_code_implements_nothing():
    """Reported on PR #117. Stripping dead function *bodies* left their imports
    behind, and `imported_modules` counts an import whether or not anything
    uses it — the same false retirement, through the import half."""
    parent = "import catboost as cb\n\n\ndef unused():\n    return cb.train()\n" + _ENTRY

    assert check_redundancy(parent, ["cb"]).redundant is False


def test_an_import_the_live_code_uses_still_counts():
    """The carve-out must not cost the behaviour it guards."""
    parent = "import catboost as cb\n\n\ndef main():\n    return cb.train()\n" + (
        '\n\nif __name__ == "__main__":\n    main()\n'
    )

    assert check_redundancy(parent, ["cb"]).redundant is True


def test_a_two_level_dead_chain_implements_nothing():
    """One strip pass removes only the leaves: an unreachable wrapper still
    references the helper it calls, so the helper looks live until the wrapper
    is gone. The rogii false retirement, one indirection deeper."""
    parent = (
        "import pandas as pd\n"
        "\n"
        "\n"
        "def _roll(df):\n"
        '    return df.groupby("p")["x"].transform(lambda s: s.rolling(5).mean())\n'
        "\n"
        "\n"
        "def engineer(df):\n"
        "    return _roll(df)\n" + _ENTRY
    )

    assert check_redundancy(parent, ["rolling", "groupby"]).redundant is False


def test_a_live_chain_still_counts():
    parent = (
        "import pandas as pd\n"
        "\n"
        "\n"
        "def _roll(df):\n"
        '    return df.groupby("p")["x"].transform(lambda s: s.rolling(5).mean())\n'
        "\n"
        "\n"
        "def engineer(df):\n"
        "    return _roll(df)\n"
        "\n"
        "\n"
        "def main():\n"
        "    engineer(None)\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )

    assert check_redundancy(parent, ["rolling", "groupby"]).redundant is True


def test_a_callback_only_helper_is_already_implemented():
    """Reported on PR #117: `check_redundancy` was the fourth consumer left on
    `called_names`, so a helper the parent only ever hands to `df.apply` read
    as *not* implemented — and the paid aider call that followed made no edit,
    raised `aider_no_edit` rather than `hypothesis_redundant`, and left the
    hypothesis to be picked again."""
    parent = (
        "import pandas as pd\n\n\n"
        "def helper(row):\n    return row['a']\n\n\n"
        "def main():\n    df = pd.read_csv('x')\n    df.apply(helper, axis=1)\n\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )

    assert check_redundancy(parent, ["helper"]).redundant is True


def test_one_unused_alias_does_not_keep_its_neighbour_alive():
    """`import pandas as pd, numpy as np` with only `pd` used trims to one
    alias but still yields one statement, so a count check called it unchanged
    and threw the trimmed version away."""
    parent = (
        "import pandas as pd, numpy as np\n\n\n"
        "def dead():\n    return np.mean\n\n\n"
        "def main():\n    return pd\n\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )

    assert check_redundancy(parent, ["numpy"]).redundant is False


def test_an_unrelated_attribute_does_not_keep_a_dead_import_alive():
    """`used` folded in every `Attribute.attr`, so an unrelated `df.time` made
    `import time` look alive after its only real user was stripped."""
    import ast

    from labpilot.research_engine.execution.delta.consistency import (
        imported_modules,
        strip_unreachable,
    )

    parent = (
        "import time\n\n\n"
        "def dead():\n    return time.time()\n\n\n"
        "def main(df):\n    return df.time\n\n\n"
        'if __name__ == "__main__":\n    main(None)\n'
    )

    assert imported_modules(strip_unreachable(ast.parse(parent))) == set()


def test_a_local_variable_does_not_satisfy_a_claim():
    """Reported on PR #117 as a regression from switching to `present_names`.

    `for rolling in range(3): print(rolling)` binds and loads a local, and a
    hypothesis proposing a real rolling-window feature read as already
    implemented — retired before aider ever ran. Load context does not separate
    those (`print(rolling)` is a load); "is this a function this module
    defines" does.
    """
    parent = (
        "import pandas as pd\n\n\n"
        "def engineer_features(df):\n"
        "    for rolling in range(3):\n"
        "        print(rolling)\n"
        "    return df\n\n\n"
        "def main():\n    engineer_features(None)\n\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )

    assert check_redundancy(parent, ["rolling"]).redundant is False


def test_a_real_rolling_call_still_satisfies_it():
    """The carve-out must not cost the behaviour it guards."""
    parent = (
        "import pandas as pd\n\n\n"
        "def engineer_features(df):\n"
        "    return df.groupby('p')['x'].rolling(5).mean()\n\n\n"
        "def main():\n    engineer_features(None)\n\n\n"
        'if __name__ == "__main__":\n    main()\n'
    )

    assert check_redundancy(parent, ["rolling"]).redundant is True
