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


# -- what happens *after* detection -------------------------------------------


class _Plan:
    def __init__(self, hypothesis_id: str = "H-001") -> None:
        self.hypothesis_id = hypothesis_id


class _Paths:
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir


class _Context:
    """Only the attributes `_propose_delta` actually reads."""

    def __init__(self, tmp_path, *, strategy: str = "delta") -> None:
        self.constraints = {"codegen_strategy": strategy}
        self.workspace_root = tmp_path
        self.plan = _Plan()
        self.paths = _Paths(tmp_path / "knowledge")
        self.competition = "demo"


def _capability_that_finds_redundancy(monkeypatch, *, retired: list[str]):
    """A capability whose aider raises `hypothesis_redundant`."""
    from labpilot.research_engine.execution.capabilities.code_engineering import capability as mod

    cap = mod.CodeEngineeringCapability(llm_client=_Gateway())

    class _Agent:
        def __init__(self, gateway):
            pass

        def propose(self, structured, parent):
            raise AiderError("already implemented: parent calls 'cb'", kind="hypothesis_redundant")

    monkeypatch.setattr("labpilot.research_engine.execution.delta.aider_agent.AiderAgent", _Agent)
    monkeypatch.setattr(
        mod.CodeEngineeringCapability,
        "_retire_redundant_hypothesis",
        lambda self, context, reason: (retired.append(reason), True)[1],
    )
    return cap, mod


def test_redundancy_stops_the_step_instead_of_falling_back(tmp_path, monkeypatch):
    """The gap the source-grep test above could not see.

    Detection retired the hypothesis and then returned `(None, "")` like any
    other decline, so the whole-file agent rewrote `train.py` and the runner
    trained it — spending the entire experiment the retirement existed to
    avoid, on work the system had just proved was already done.
    """
    retired: list[str] = []
    cap, mod = _capability_that_finds_redundancy(monkeypatch, retired=retired)

    with pytest.raises(mod.RedundantHypothesisError):
        cap._propose_delta(_Context(tmp_path), _Ctx(), "print('parent')")

    assert retired, "the hypothesis must still be retired when the step fails"


def test_other_aider_failures_still_fall_back(tmp_path, monkeypatch):
    """The carve-out is narrow on purpose. Every other decline is a codegen
    problem, where the experiment is still worth running — §10 requires both
    paths to coexist while the failure rate is measured."""
    from labpilot.research_engine.execution.capabilities.code_engineering import capability as mod

    cap = mod.CodeEngineeringCapability(llm_client=_Gateway())

    class _Agent:
        def __init__(self, gateway):
            pass

        def propose(self, structured, parent):
            raise AiderError("aider made no edit", kind="aider_no_edit")

    monkeypatch.setattr("labpilot.research_engine.execution.delta.aider_agent.AiderAgent", _Agent)

    assert cap._propose_delta(_Context(tmp_path), _Ctx(), "print('parent')") == (None, "")


def test_a_failed_retirement_is_named_in_the_error(tmp_path, monkeypatch):
    """Swallowing a store failure left the hypothesis `proposed` while the
    caller behaved as though it were retired — the loop this whole path exists
    to break, reachable through one uncaught store error."""
    from labpilot.research_engine.execution.capabilities.code_engineering import capability as mod

    cap = mod.CodeEngineeringCapability(llm_client=_Gateway())

    class _Agent:
        def __init__(self, gateway):
            pass

        def propose(self, structured, parent):
            raise AiderError("already implemented", kind="hypothesis_redundant")

    monkeypatch.setattr("labpilot.research_engine.execution.delta.aider_agent.AiderAgent", _Agent)
    monkeypatch.setattr(
        mod.CodeEngineeringCapability,
        "_retire_redundant_hypothesis",
        lambda self, context, reason: False,
    )

    with pytest.raises(mod.RedundantHypothesisError, match="could not be retired"):
        cap._propose_delta(_Context(tmp_path), _Ctx(), "print('parent')")


def test_redundancy_reads_the_whole_parent_not_the_clipped_copy(tmp_path):
    """`prior_train_py` is clipped to 120k by the capability. `ast.parse` on a
    mid-statement clip raises `SyntaxError`, redundancy.py declines to judge,
    and the verdict comes back "not redundant" — silently, and only for
    pipelines large enough for the problem to matter."""
    import ast

    # An open brace, so the clip lands inside a literal and the parse fails.
    # Comment padding would survive being cut and the test would pass against
    # the bug it is written to catch — checked, not assumed.
    entries = "\n".join(f'    "feature_{i}_{"x" * 20}": {i},' for i in range(6000))
    big = _ENSEMBLE + "\nFEATURES = {\n" + entries + "\n}\n"
    assert len(big) > 120_000
    with pytest.raises(SyntaxError):
        ast.parse(big[:120_000])
    ast.parse(big)

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(big, encoding="utf-8")
    agent = _agent(DeltaBrief(instruction="ensemble them", added=["lgb", "DecisionTreeRegressor"]))

    with pytest.raises(AiderError) as caught:
        agent.propose(_Ctx(prior_train_py=big[:120_000]), tmp_path)

    assert caught.value.kind == "hypothesis_redundant"


# --- a retry asks for the repair, not the hypothesis again -------------------


def test_the_instruction_leads_with_the_error_on_a_retry():
    """Two stalls on rogii 2026-08-09 came from here: `retry_reason` reached
    the whole-file prompt and the delta brief, but never aider's instruction,
    so a retry re-sent the same hypothesis and the editor declined."""
    from labpilot.research_engine.execution.delta.aider_agent import _instruction

    text = _instruction(_Ctx(plan_goal="add rolling features", retry_reason="KeyError: 'TVT'"))

    assert "KeyError: 'TVT'" in text
    assert text.index("repair") < text.index("add rolling features")


def test_the_hypothesis_is_still_carried_so_the_fix_preserves_it():
    """The failure being repaired is usually *in* the change, so a repair with
    no context can revert the experiment instead of fixing it."""
    from labpilot.research_engine.execution.delta.aider_agent import _instruction

    text = _instruction(_Ctx(plan_goal="add rolling features", retry_reason="KeyError: 'TVT'"))

    assert "add rolling features" in text
    assert "Preserve" in text


def test_without_a_failure_it_is_the_hypothesis_alone():
    """The common case must not gain repair language it has no reason to."""
    from labpilot.research_engine.execution.delta.aider_agent import _instruction

    text = _instruction(_Ctx(plan_goal="add rolling features", prediction="MSE drops"))

    assert "repair" not in text
    assert "add rolling features" in text


def test_a_retry_overrides_the_brief_instruction(tmp_path):
    """The brief turns a hypothesis into an instruction plus a claim. A retry
    has neither to offer — the change is already made — so leaving the brief in
    charge kept re-asking for the technique, and aider kept declining."""
    seen: list[str] = []

    def _runner(cmd, cwd, timeout):
        seen.append(" ".join(str(c) for c in cmd))
        raise RuntimeError("stop here")

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("x = 1\n", encoding="utf-8")
    agent = _agent(DeltaBrief(instruction="add the MD_x_GR feature"), runner=_runner)

    with pytest.raises(Exception):
        agent.propose(
            _Ctx(prior_train_py="x = 1\n", retry_reason="KeyError: 'TVT'", plan_goal="add MD_x_GR"),
            tmp_path,
        )

    assert seen, "aider should still be spawned"
    assert "KeyError" in seen[0]


def test_a_repair_carries_the_brief_as_well_as_the_error(tmp_path):
    """Reported on PR #117: the brief call is paid for and its instruction was
    discarded on every retry. It rides along now — after the error, so a brief
    that ignored the retry cannot leave the failure unmentioned."""
    seen: list[str] = []

    def _runner(cmd, cwd, timeout):
        seen.append(" ".join(str(c) for c in cmd))
        raise RuntimeError("stop here")

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(_ENSEMBLE, encoding="utf-8")
    agent = _agent(DeltaBrief(instruction="add the MD_x_GR feature"), runner=_runner)

    with pytest.raises(Exception):
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE, retry_reason="KeyError: 'TVT'"), tmp_path)

    sent = seen[0]
    assert "KeyError" in sent
    assert "MD_x_GR" in sent
    assert sent.index("KeyError") < sent.index("MD_x_GR")


def test_a_repair_is_not_judged_redundant(tmp_path):
    """Reported on PR #117: `check_redundancy` ran before `retrying` was even
    computed, so a retry whose parent legitimately contained the earlier
    attempt's code was retired instead of repaired."""
    spawned: list[str] = []

    def _runner(cmd, cwd, timeout):
        spawned.append("spawned")
        raise RuntimeError("stop here")

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text(_ENSEMBLE, encoding="utf-8")
    agent = _agent(DeltaBrief(instruction="ensemble them", added=["lgb"]), runner=_runner)

    with pytest.raises(Exception):
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE, retry_reason="KeyError: 'TVT'"), tmp_path)

    assert spawned == ["spawned"], "a repair must reach aider, not be retired"


def test_without_a_retry_reason_redundancy_still_retires(tmp_path):
    agent = _agent(DeltaBrief(instruction="ensemble them", added=["lgb"]))

    with pytest.raises(AiderError) as caught:
        agent.propose(_Ctx(prior_train_py=_ENSEMBLE), tmp_path)

    assert caught.value.kind == "hypothesis_redundant"
