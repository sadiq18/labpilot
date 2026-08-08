"""Observe-only delta recording on the whole-file codegen path (M19 step 1b).

Deliberately partial. Only the checks needing **no vocabulary** run: what the
change touched, and whether it was wide. `keep`/`add`/`combine` are not derived,
because nothing maps a technique name to a code identifier — see
`_observe_delta`'s docstring for the rogii measurement that killed that idea.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.capabilities.code_engineering.capability import (
    _observe_delta,
)
from labpilot.research_engine.execution.schemas.code_proposal import (
    CodeFileSpec,
    CodeProposal,
)

PARENT = """
def build_features(frame):
    return frame


def train(X, y):
    return X
"""

TOUCHED_ONE = """
def build_features(frame):
    return frame


def train(X, y):
    scaled = X * 2
    return scaled
"""


def _proposal(content: str, **claim) -> CodeProposal:
    return CodeProposal(files=[CodeFileSpec(path="pipeline/train.py", content=content)], **claim)


# --- the agent declares its own claim, in code identifiers ------------------

_PARENT_LGB = """
import lightgbm as lgb


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    return a.predict(X)
"""

_SUBSTITUTED = """
import catboost as cb


def train(X, y):
    b = cb.CatBoostRegressor()
    b.fit(X, y)
    return b.predict(X)
"""

_ADDED_UNUSED = """
import lightgbm as lgb
import catboost as cb


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    b = cb.CatBoostRegressor()
    b.fit(X, y)
    return a.predict(X)
"""

_ENSEMBLED = """
import lightgbm as lgb
import catboost as cb


def train(X, y):
    a = lgb.LGBMRegressor()
    a.fit(X, y)
    b = cb.CatBoostRegressor()
    b.fit(X, y)
    return 0.5 * a.predict(X) + 0.5 * b.predict(X)
"""

_CLAIM = {"kept": ["lgb"], "added": ["cb"], "combined": ["lgb", "cb"]}


def test_a_substitution_contradicts_the_claim():
    """The card would read "ensembling improved MSE" for a run that measured
    substitution."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_SUBSTITUTED, **_CLAIM))
    assert meta["delta_consistent"] is False
    assert any("should have been kept" in v for v in meta["delta_violations"])


def test_a_discarded_second_model_contradicts_the_claim():
    """The quietest failure: the constructor is present, so addition passes,
    but the score reflects the parent alone."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_ADDED_UNUSED, **_CLAIM))
    assert meta["delta_consistent"] is False
    assert any("no aggregation" in v for v in meta["delta_violations"])


def test_an_honest_ensemble_passes():
    """A check that rejects everything is a blocker, not a check. Uses a
    weighted blend — the standard technique, which calls no aggregator."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_ENSEMBLED, **_CLAIM))
    assert meta["delta_consistent"] is True, meta["delta_violations"]


def test_the_claim_is_recorded_even_when_it_holds():
    """A reader has to be able to see what was claimed, not just whether it
    passed — the verdict alone cannot be audited against the hypothesis."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_ENSEMBLED, **_CLAIM))
    assert meta["delta_claim"] == {
        "kept": ["lgb"],
        "added": ["cb"],
        "combined": ["lgb", "cb"],
    }


def test_a_baseline_claiming_nothing_gets_no_verdict():
    """Empty is honest for a from-scratch baseline. Reporting `consistent:
    true` would be a pass nobody earned."""
    meta = _observe_delta("", _proposal(_ENSEMBLED))
    assert "delta_consistent" not in meta
    assert meta["delta_claim"] == {"kept": [], "added": [], "combined": []}


def test_observing_never_blocks_regardless_of_the_verdict():
    """Observe-only: these checks have only ever been calibrated against
    hand-written samples, which is exactly how the step 1a bugs got in."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_SUBSTITUTED, **_CLAIM))
    assert meta["delta_consistent"] is False
    assert isinstance(meta, dict)  # returned, not raised


def test_it_records_which_functions_changed():
    meta = _observe_delta(PARENT, _proposal(TOUCHED_ONE))
    assert meta["delta_touched_functions"] == ["train"]


def test_it_never_claims_consistency_it_did_not_check():
    """No claim was supplied, so reporting `consistent: true` would be a pass
    nobody earned — the same fabricated-verdict failure as the placeholder
    cards and the inverted metric direction."""
    meta = _observe_delta(PARENT, _proposal(TOUCHED_ONE))
    assert "delta_consistent" not in meta
    assert "delta_violations" not in meta


def test_a_narrow_delta_carries_no_wide_delta_flag():
    """Specifically the *confinement* flag. A claimless delta also carries the
    "unchecked" flag, which is a different statement — about what was not
    verified, not about how wide the change was."""
    flags = _observe_delta(PARENT, _proposal(TOUCHED_ONE, added=["scaled"]))["delta_flags"]
    assert not any("touches" in f for f in flags)


def test_a_wide_delta_is_flagged():
    """Measured against rogii's real 331-line train.py: 8 functions, so the
    threshold of 5 fires at 6+ — a targeted technique delta stays silent while
    a genuine whole-file rewrite flags."""
    wide = PARENT + "\n".join(f"def f{i}(x):\n    return x + {i}\n" for i in range(8))
    meta = _observe_delta(PARENT, _proposal(wide))
    assert meta["delta_flags"], "a wide delta must be recorded on the card"
    assert "attribution" in meta["delta_flags"][0]


def test_a_baseline_with_no_parent_records_nothing_spurious():
    """A baseline has no parent to diff against, so there is no delta to
    describe — not a delta of everything."""
    meta = _observe_delta("", _proposal(TOUCHED_ONE))
    assert meta["delta_touched_functions"] == []


def test_a_proposal_without_train_py_is_skipped():
    proposal = CodeProposal(files=[CodeFileSpec(path="pipeline/infer.py", content="x = 1\n")])
    assert _observe_delta(PARENT, proposal) == {}


def test_unparseable_generated_code_does_not_crash_the_run():
    """Observe-only means observe-only: a syntax error is the apply path's
    problem, and recording must not raise on the way there."""
    meta = _observe_delta(PARENT, _proposal("def broken(:\n"))
    assert isinstance(meta, dict)


def test_reformatting_is_not_recorded_as_a_change():
    """Compared on the AST, so a whole-file regeneration that only reflows the
    parent does not look like a delta touching everything."""
    reflowed = PARENT.replace("def train(X, y):", "def train(X,   y):  # regenerated")
    assert _observe_delta(PARENT, _proposal(reflowed))["delta_touched_functions"] == []


# --- the claim must survive how models actually answer ----------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({}, []),
        ({"kept": None}, []),
        ({"kept": []}, []),
        ({"kept": "lgb"}, ["lgb"]),
        ({"kept": ""}, []),
        ({"kept": ["lgb", "cb"]}, ["lgb", "cb"]),
    ],
)
def test_a_claim_parses_however_the_model_spells_it(payload, expected):
    """`null` means "nothing", not "invalid".

    Models emit `"kept": null` about as readily as `[]`. Rejecting it raises a
    ValidationError, which the retry path reads as a malformed response and
    re-asks — burning a step from a 30-step campaign over optional metadata.
    """
    import json

    assert CodeProposal.model_validate_json(json.dumps(payload)).kept == expected


def test_a_helper_that_is_defined_but_never_called_is_a_violation():
    """`check_addition` asks whether the symbol is *called or imported*, not
    whether it was *defined*. A function nothing calls changes no behaviour, so
    crediting a technique for it would be a false attribution — the same
    "added but unused" failure as a discarded second model.

    Found by running the real rogii train.py through this helper: a declared
    `_blend` that was defined and never wired in reported inconsistent, which
    is correct.
    """
    parent = "def train(X):\n    return X\n"
    child = "def _blend(a, b):\n    return (a + b) / 2\n\n\ndef train(X):\n    return X\n"
    meta = _observe_delta(parent, _proposal(child, added=["_blend"]))
    assert meta["delta_consistent"] is False


def test_a_helper_that_is_defined_and_used_passes():
    parent = "def train(X):\n    return X\n"
    child = (
        "def _blend(a, b):\n    return (a + b) / 2\n\n\ndef train(X):\n    return _blend(X, X)\n"
    )
    meta = _observe_delta(parent, _proposal(child, added=["_blend"]))
    assert meta["delta_consistent"] is True, meta["delta_violations"]


# --- an unchecked delta must not look like a clean one ----------------------


def test_a_delta_that_declares_nothing_is_marked_unchecked():
    """Without this, a delta that skipped the claim and one that passed every
    check are indistinguishable on the card: both show no verdict and no
    violations. An unchecked experiment would read as a clean one."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_ENSEMBLED))
    assert meta["delta_unchecked"] is True
    assert meta["delta_claim_declared"] is False
    assert any("declared no kept/added/combined" in f for f in meta["delta_flags"])


def test_a_baseline_that_declares_nothing_is_not_marked_unchecked():
    """A baseline claims nothing because there is nothing to claim — there is
    no parent to preserve anything from."""
    meta = _observe_delta("", _proposal(_ENSEMBLED))
    assert "delta_unchecked" not in meta
    assert meta["delta_claim_declared"] is False
    assert meta["delta_flags"] == []


def test_a_declared_delta_is_not_marked_unchecked():
    meta = _observe_delta(_PARENT_LGB, _proposal(_ENSEMBLED, **_CLAIM))
    assert "delta_unchecked" not in meta
    assert meta["delta_claim_declared"] is True


def test_the_missing_claim_is_recorded_not_refused():
    """Observe-only holds here too: the rate is what is worth knowing first.
    If codegen routinely omits the claim that is a prompt problem to fix with a
    number attached, not a reason to fail runs today."""
    meta = _observe_delta(_PARENT_LGB, _proposal(_ENSEMBLED))
    assert isinstance(meta, dict)
    assert "delta_consistent" not in meta
