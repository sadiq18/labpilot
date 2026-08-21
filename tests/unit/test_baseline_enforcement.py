"""M23 step 8: no hypothesis minting until the gate passes.

**Enforcement is at hypothesis generation, not at submission.** A campaign whose
pipeline loses to a constant may still run plans, implement and reflect — those
are how the gate gets *opened* — but it may not mint hypotheses, because that is
where a false belief enters the store and outlives the run.

rogii's cost was 19 child hypotheses and eight techniques driven to 0.0
confidence, all written down while the pipeline was 91x worse than one line of
code. **None of it was a submission.** Gating `submit` would have caught none of
it; gating `run_plan` would have stopped the campaign from ever opening the gate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from labpilot.research_engine.execution.baseline.gate import (
    Waiver,
    evaluate_gate,
    reading_fingerprint,
    refuse_hypothesis_minting,
    write_waiver,
)
from labpilot.research_engine.execution.baseline.runner import ensure_readings


def _workspace(tmp_path: Path, *, learnable: bool, modality: str = "tabular") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    y = 3 * x1 + rng.normal(0, 0.3, n) if learnable else rng.normal(size=n)
    pd.DataFrame({"x1": x1, "x2": rng.normal(size=n), "y": y}).to_csv(
        tmp_path / "train.csv", index=False
    )
    (tmp_path / "baseline_choice.json").write_text(
        json.dumps(
            {
                "problem_type": "tabular_regression",
                "template_name": "t",
                "rationale": "",
                "metric_name": "rmse",
                "target_column": "y",
                "train_file": "train.csv",
                "validation": {"scheme": "kfold", "n_splits": 4},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "profile.json").write_text(
        json.dumps(
            {
                "competition": "demo",
                "schema_version": 4,
                "target_column": "y",
                "row_count": n,
                "train_file": "train.csv",
                "modalities": [
                    {"modality": modality, "present": True, "role": "primary", "confidence": 0.9}
                ],
                "columns": [
                    {"name": "x1", "dtype": "float64", "unique_count": n, "is_numeric": True},
                    {"name": "x2", "dtype": "float64", "unique_count": n, "is_numeric": True},
                    {
                        "name": "y",
                        "dtype": "float64",
                        "unique_count": n,
                        "is_numeric": True,
                        "stats": {"min": -9.0, "max": 9.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ensure_readings(tmp_path)
    return tmp_path


# --- the refusal itself ---------------------------------------------------------


def test_a_failing_gate_refuses_minting_when_enforced(tmp_path: Path) -> None:
    _workspace(tmp_path, learnable=False)
    assert evaluate_gate(tmp_path).state == "failed"

    refusal = refuse_hypothesis_minting(tmp_path, enforced=True)

    assert refusal
    assert "failed" in refusal


def test_observe_only_refuses_nothing(tmp_path: Path) -> None:
    """The default, and the whole shape of the rollout: step 8 is a config flip.

    A failing gate is recorded and reported, and the campaign carries on, until
    a campaign's worth of verdicts turns "the gate is right" into a measured
    false-positive rate.
    """
    _workspace(tmp_path, learnable=False)

    assert refuse_hypothesis_minting(tmp_path, enforced=False) == ""


def test_a_passing_gate_refuses_nothing(tmp_path: Path) -> None:
    _workspace(tmp_path, learnable=True)
    assert evaluate_gate(tmp_path).state == "passed"

    assert refuse_hypothesis_minting(tmp_path, enforced=True) == ""


def test_a_waiver_reopens_minting(tmp_path: Path) -> None:
    """Someone accepted `failed` in writing, and it is recorded against the
    fingerprint it was granted for."""
    _workspace(tmp_path, learnable=False)
    write_waiver(
        tmp_path,
        Waiver(reason="known-bad, shipping anyway", fingerprint=reading_fingerprint(tmp_path)),
    )

    assert refuse_hypothesis_minting(tmp_path, enforced=True) == ""


def test_no_workspace_is_answered_quietly(caplog: pytest.LogCaptureFixture) -> None:
    """A caller with no workspace is one the gate cannot see, not one that
    skipped its baseline.

    The assertion is on the *silence*. Without the explicit guard, `Path(None)`
    raises and the exception handler returns the same empty string — so two
    mutations of this function produced one answer and neither an empty-return
    check nor a "the gate was never called" check could tell them apart. What
    does differ is that the fallback logs `Baseline gate could not be evaluated`
    on every path that never had a workspace, which is a warning an operator
    would learn to ignore and would then miss when it meant something.
    """
    caplog.set_level(logging.WARNING)

    assert refuse_hypothesis_minting(None, enforced=True) == ""

    assert "could not be evaluated" not in caplog.text


def test_a_gate_that_cannot_run_does_not_block(tmp_path: Path) -> None:
    """A fault here would read as "baseline not passed", which is exactly the
    failure mode `H-BASELINE.status` was rejected for."""
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module

    _workspace(tmp_path, learnable=False)
    with mock.patch.object(gate_module, "evaluate_gate", side_effect=RuntimeError("boom")):
        assert refuse_hypothesis_minting(tmp_path, enforced=True) == ""


def test_the_refusal_names_a_next_step(tmp_path: Path) -> None:
    """A refusal an operator cannot act on is a wall, and each of the nine
    states has a different thing to do about it."""
    _workspace(tmp_path, learnable=False)

    refusal = refuse_hypothesis_minting(tmp_path, enforced=True)

    assert "does not beat a constant" in refusal


# --- the durable write ------------------------------------------------------------


def _recommendation():
    from labpilot.research_engine.intelligence.hypothesis.models import HypothesisRecommendation

    return HypothesisRecommendation(
        rank=1, hypothesis_id="", title="try more trees", observation="o", reason="r"
    )


def test_nothing_reaches_the_store_when_the_gate_refuses(tmp_path: Path) -> None:
    """`persist_recommendations` is the only durable write, and it covers the
    CLI path too. Blocking further upstream would stop the campaign doing the
    work that opens the gate.

    Asserted on the **store**, not on the source. The first version of this test
    grepped `persist.py` for the call, which a mutation turning `if refusal:`
    into `if False:` sailed straight through — it proved the line was written,
    never that it did anything.
    """
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.intelligence.hypothesis.persist import persist_recommendations
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    workspace = _workspace(tmp_path / "ws", learnable=False)
    knowledge = tmp_path / "knowledge"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        written = persist_recommendations(
            [_recommendation()],
            knowledge_dir=knowledge,
            competition="demo",
            workspace_root=workspace,
        )

    assert written == []
    assert HypothesisStore(knowledge, "demo").list() == [], "nothing durable was created"


def test_the_store_write_proceeds_when_the_gate_passes(tmp_path: Path) -> None:
    """The refusal has to be able to *not* fire, or it is a wall."""
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.intelligence.hypothesis.persist import persist_recommendations
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore

    workspace = _workspace(tmp_path / "ws", learnable=True)
    knowledge = tmp_path / "knowledge"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        written = persist_recommendations(
            [_recommendation()],
            knowledge_dir=knowledge,
            competition="demo",
            workspace_root=workspace,
        )

    assert len(written) == 1 and written[0].hypothesis_id
    assert len(HypothesisStore(knowledge, "demo").list()) == 1


def test_the_analyze_path_hands_over_its_workspace() -> None:
    """A gate the caller never gives a root to is a gate that never fires."""
    import inspect

    from labpilot.research_engine.intelligence import orchestrator

    assert "workspace_root=context.workspace_root" in inspect.getsource(orchestrator)


# --- goal 4, tested against a campaign rather than against a file --------------------


def test_an_unchangeable_state_does_not_pin_the_campaign_to_baseline(tmp_path: Path) -> None:
    """Review finding, and the previous goal-4 test could not have caught it.

    An image dataset reaches `awaiting_ml` — Baseline 1 cannot run where
    features are not columns, and no re-run changes that. Answering "the
    baseline is not done" forever left `generate_plan` pinned to `baseline`, and
    because baseline compilation is idempotent the campaign recompiled the same
    plan and could never run a second experiment. `actions.py`'s own docstring
    describes exactly that failure.

    This is the trap the design names — *"a gate demanding something
    unaffordable gets disabled"* — arriving through the door goal 4 exists to
    bolt.
    """
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.conductor.actions import resolve_step_args
    from labpilot.research_engine.conductor.loop import _baseline_is_done

    workspace = _workspace(tmp_path / "ws", learnable=False, modality="image")

    class _Workspace:
        root = workspace
        knowledge_dir = tmp_path / "knowledge"
        competition = "demo"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        assert evaluate_gate(workspace).state == "awaiting_ml"
        settled = _baseline_is_done(_Workspace())

    assert settled, "nothing more can be done about the baseline here"
    args = resolve_step_args(
        "generate_plan",
        {"baseline": True},
        latest_plan_id="P-001",
        latest_execution_id="E-001",
        next_hypothesis_id="H-002",
        baseline_plan_exists=settled,
    )
    assert args == {"hypothesis_id": "H-002"}, "the campaign moves on to iterating"


def test_a_failing_gate_still_pins_the_campaign_to_baseline(tmp_path: Path) -> None:
    """The other half. `failed` is fixable by fixing the pipeline, so the
    campaign keeps being told to produce a baseline that beats a constant."""
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.conductor.actions import resolve_step_args
    from labpilot.research_engine.conductor.loop import _baseline_is_done

    workspace = _workspace(tmp_path / "ws", learnable=False)

    class _Workspace:
        root = workspace
        knowledge_dir = tmp_path / "knowledge"
        competition = "demo"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        assert evaluate_gate(workspace).state == "failed"
        settled = _baseline_is_done(_Workspace())

    assert not settled
    args = resolve_step_args(
        "generate_plan",
        {"baseline": True},
        latest_plan_id="P-001",
        latest_execution_id="E-001",
        next_hypothesis_id="H-002",
        baseline_plan_exists=settled,
    )
    assert args == {"baseline": True}, "still asking for a baseline"


def test_an_unchangeable_state_does_not_refuse_minting(tmp_path: Path) -> None:
    """A gate that refuses forever on a property of the dataset is one an
    operator switches off, and then it protects nothing at all."""
    workspace = _workspace(tmp_path / "ws", learnable=False, modality="image")

    assert evaluate_gate(workspace).state == "awaiting_ml"
    assert refuse_hypothesis_minting(workspace, enforced=True) == ""


def test_the_three_questions_are_answered_separately() -> None:
    """ "The gate is not open", "nothing more can be done", and "a belief written
    now would be unsafe" are three facts, and conflating them is what pinned the
    campaign."""
    from labpilot.research_engine.execution.baseline.gate import (
        baseline_is_settled,
        blocks_research,
        refuses_minting,
    )

    assert blocks_research("awaiting_ml"), "the gate is not open"
    assert baseline_is_settled("awaiting_ml"), "and there is nothing to do about it"
    assert not refuses_minting("awaiting_ml"), "so it must not refuse forever"

    assert blocks_research("failed")
    assert not baseline_is_settled("failed")
    assert refuses_minting("failed")


# --- goal 4: the allowlist never empties -------------------------------------------


def test_run_plan_survives_a_closed_gate() -> None:
    """Gating `run_plan` would be a bug: it is how the gate gets opened.

    With the gate closed there must still be a tool that can open it, or a
    campaign that fails its baseline can never do anything about it — and a gate
    nobody can pass is one everybody switches off.
    """
    import inspect

    from labpilot.research_engine.execution.baseline import gate as gate_module

    source = inspect.getsource(gate_module)

    for tool in ("run_plan", "implement", "reflect", "generate_plan"):
        assert f'"{tool}"' not in source, f"{tool} must not be gated here"


def test_only_hypothesis_minting_is_gated() -> None:
    """The refusal has exactly one production caller, and it is the store write."""
    import subprocess

    out = subprocess.run(
        ["grep", "-rn", "refuse_hypothesis_minting", "--include=*.py", "src"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout

    # The import line is not a call site; counting it made "exactly one caller"
    # read as two, and the assertion then said nothing about what it meant.
    callers = [
        line
        for line in out.splitlines()
        if "gate.py" not in line and "def " not in line and "import" not in line
    ]
    assert len(callers) == 1, callers
    assert "persist.py" in callers[0]


# --- the plan-file question, retired ------------------------------------------------


def test_a_compiled_plan_is_not_a_finished_baseline(tmp_path: Path) -> None:
    """`_baseline_plan_exists` answered "was a plan object compiled?" to a caller
    asking "has the baseline been done?".

    A campaign flipped to research mode on the strength of a file existing,
    whatever the pipeline it described actually scored — which is how rogii spent
    two weeks minting hypotheses over a pipeline 91x worse than one line of code.
    """
    import inspect

    from labpilot.research_engine.conductor import loop

    source = inspect.getsource(loop)

    assert "baseline_plan_exists=_baseline_is_done(workspace)" in source
    assert source.count("baseline_plan_exists=_baseline_plan_exists(workspace)") == 0
    assert "enforcement_enabled()" in inspect.getsource(loop._baseline_is_done)


@pytest.mark.parametrize("enforced", [False, True])
def test_the_plan_lookup_remains_the_observe_only_answer(tmp_path: Path, enforced: bool) -> None:
    """Observe-only must not change what a campaign does, so the old answer
    stands until the flip."""
    import inspect

    from labpilot.research_engine.conductor import loop

    source = inspect.getsource(loop._baseline_is_done)

    assert "return _baseline_plan_exists(workspace)" in source
    if enforced:
        assert "baseline_is_settled" in source
