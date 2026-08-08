"""Redundancy is detected on the live path, not merely detectable.

Both halves of this fix existed as functions before anything called them, which
is the failure `00-diagnosis.md` opens with — *"every milestone shipped its
structure but not its function"*. These tests exist to keep that from being true
of this one.

The property: a hypothesis whose change the parent already implements is
retired **before** aider is spawned, and is therefore never selected again.
"""

from __future__ import annotations

import pytest

from labpilot.research_engine.execution.delta.aider_agent import AiderAgent, AiderError
from labpilot.research_engine.execution.schemas.delta_brief import DeltaBrief

_ENSEMBLE = """
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor

def train(X, y):
    a = lgb.train(X, y)
    b = DecisionTreeRegressor().fit(X, y).predict(X)
    return (a + b) / 2
"""


class _Gateway:
    def for_role(self, role):  # pragma: no cover - identity only
        return self


class _Ctx:
    def __init__(self, **data):
        self.competition = "rogii-wellbore-geology-prediction"
        self.data = data
        self.question = ""
        self.text = ""


def _agent(brief: DeltaBrief, runner=None) -> AiderAgent:
    class _Brief:
        def run(self, ctx):
            return brief

    def _explode(cmd, cwd, timeout):  # pragma: no cover - must not be reached
        raise AssertionError("aider was spawned for a redundant hypothesis")

    return AiderAgent(_Gateway(), runner=runner or _explode, brief_agent=_Brief())


def test_a_redundant_hypothesis_is_refused_before_aider_runs(tmp_path):
    """The runner asserts if called, so reaching it fails the test — the saving
    is the point, not a side effect."""
    agent = _agent(DeltaBrief(instruction="ensemble them", added=["lgb", "DecisionTreeRegressor"]))

    with pytest.raises(AiderError) as caught:
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert caught.value.kind == "hypothesis_redundant"


def test_the_refusal_names_the_symbol_that_proves_it(tmp_path):
    agent = _agent(DeltaBrief(added=["lgb"]))

    with pytest.raises(AiderError) as caught:
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert "'lgb'" in str(caught.value)


def test_redundancy_is_distinct_from_aider_failing(tmp_path):
    """`aider_no_edit` and `hypothesis_redundant` are opposite findings — one
    says the adapter failed, the other that the campaign chose work already
    done. Sharing a kind would make step 2 read a redundancy rate as an adapter
    failure rate and conclude delta does not work."""
    agent = _agent(DeltaBrief(added=["lgb"]))

    with pytest.raises(AiderError) as caught:
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert caught.value.kind != "aider_no_edit"


def test_a_genuinely_new_symbol_reaches_aider(tmp_path):
    """The carve-out must not cost the behaviour it guards: the experiment that
    *should* run must still run."""
    reached: list[str] = []

    def _runner(cmd, cwd, timeout):
        reached.append("spawned")
        raise RuntimeError("stop here — reaching the runner is what we assert")

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(_ENSEMBLE, encoding="utf-8")
    agent = _agent(DeltaBrief(added=["CatBoostRegressor"]), runner=_runner)

    with pytest.raises(Exception):
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert reached == ["spawned"]


def test_an_empty_brief_does_not_retire_the_hypothesis(tmp_path):
    """`DeltaBriefAgent` soft-fails to an empty brief. Reading that as
    "already done" would retire good hypotheses whenever the brief model was
    unavailable."""
    reached: list[str] = []

    def _runner(cmd, cwd, timeout):
        reached.append("spawned")
        raise RuntimeError("stop here")

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(_ENSEMBLE, encoding="utf-8")
    agent = _agent(DeltaBrief(), runner=_runner)

    with pytest.raises(Exception):
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert reached == ["spawned"]


def test_the_capability_retires_on_that_kind():
    """The other half of the wiring: detection must cause retirement."""
    import inspect

    from labpilot.research_engine.execution.capabilities.code_engineering import capability

    source = inspect.getsource(capability.CodeEngineeringCapability._propose_delta)
    assert "hypothesis_redundant" in source
    assert "_retire_redundant_hypothesis" in source
