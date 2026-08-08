"""Observe-only delta recording on the whole-file codegen path (M19 step 1b).

Deliberately partial. Only the checks needing **no vocabulary** run: what the
change touched, and whether it was wide. `keep`/`add`/`combine` are not derived,
because nothing maps a technique name to a code identifier — see
`_observe_delta`'s docstring for the rogii measurement that killed that idea.
"""

from __future__ import annotations

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


def _proposal(content: str) -> CodeProposal:
    return CodeProposal(files=[CodeFileSpec(path="pipeline/train.py", content=content)])


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


def test_a_narrow_delta_carries_no_flag():
    assert _observe_delta(PARENT, _proposal(TOUCHED_ONE))["delta_flags"] == []


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
