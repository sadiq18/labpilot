"""An experiment that measures its own control is not an experiment.

Measured on rogii 2026-08-12. E-244 (H-020) and E-246 (H-021) returned
`cv_rmse` 1789.6796883967336 — the same value to sixteen digits, the same
31-feature list, the same `holdout_fraction` — with a real aider edit and a real
twelve-second training run between them. Both were recorded `succeeded`, and the
reflection chain ran over the second as though it had learned something about
H-021.

`write_code` already refuses codegen that produces *no* files, for this reason,
with the same measurement behind it: "194.80 identically because each got the
same rendered file. The run looked healthy and tested nothing." The hole left
open was codegen that produces a file which happens to be the one already there.

Two gates, because there are two ways to measure the control twice:

* the applied file is byte-identical to the parent's — caught before a training
  run is spent on it;
* the file differs but the result does not, which no digest can see, and which
  is a real finding ("this change does nothing") rather than evidence about the
  hypothesis that asked for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers.baseline_campaign import codegen_returning

from labpilot.research_engine.evidence.compare_service import compared_against_itself
from labpilot.research_engine.execution.capabilities.code_engineering import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.planner.schemas.task_types import TaskType

PARENT = (
    "def main():\n"
    "    model = fit(depth=6)\n"
    "    return model\n\n\n"
    'if __name__ == "__main__":\n'
    "    main()\n"
)
TREATMENT = PARENT.replace("depth=6", "depth=12")


def _write_code(tmp_path: Path, returns: str, *, parent: str | None, dry_run: bool = False):
    """Run WRITE_CODE with `returns` as the proposal, over an existing `parent`."""
    from test_engineer_capabilities import _ctx  # the established context factory

    context = _ctx(
        tmp_path / "knowledge",
        task_type=TaskType.WRITE_CODE,
        constraints={"dry_run": dry_run},
    )
    # `write_code` refuses to run before `prepare_workspace` has profiled the
    # data, which is a different gate and not the one under test.
    (context.workspace_root / "profile.json").write_text(
        json.dumps(
            {
                "competition": "demo",
                "files": ["train.csv"],
                "train_file": "train.csv",
                "test_file": "test.csv",
                "sample_submission_file": "sample_submission.csv",
                "row_count": 100,
                "column_count": 2,
                "columns": [
                    {"name": "id", "dtype": "int64", "unique_count": 100, "is_numeric": True},
                    {"name": "target", "dtype": "int64", "unique_count": 2, "is_numeric": True},
                ],
                "target_column": "target",
                "id_column": "id",
                "submission_columns": ["id", "target"],
                "modality": "tabular",
            }
        ),
        encoding="utf-8",
    )
    pipeline = context.workspace_root / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    if parent is not None:
        (pipeline / "train.py").write_text(parent, encoding="utf-8")

    capability = CodeEngineeringCapability()
    capability._agent = codegen_returning(returns)  # noqa: SLF001 — the injection point
    return capability.execute(context)


# --- gate one: the file itself ----------------------------------------------


def test_a_proposal_that_changes_nothing_is_refused(tmp_path: Path) -> None:
    """The gate. Without it this is a successful step that measures the parent."""
    result = _write_code(tmp_path, PARENT, parent=PARENT)

    assert result.passed is False
    assert "differs_from_parent" in result.checks
    assert "byte-identical" in (result.error or "")


def test_a_real_change_still_passes(tmp_path: Path) -> None:
    """The behaviour the gate must not cost: one line different is an experiment."""
    result = _write_code(tmp_path, TREATMENT, parent=PARENT)

    assert result.passed is True, result.error


def test_a_first_write_has_no_parent_to_match(tmp_path: Path) -> None:
    """A baseline writes `train.py` where there was none. That is not a no-op."""
    result = _write_code(tmp_path, PARENT, parent=None)

    assert result.passed is True, result.error


def test_a_dry_run_only_checks_wiring(tmp_path: Path) -> None:
    """Same carve-out the last-resort stub has: a dry run asserts the plumbing,
    not the science, and every dry run writes the same file twice by design."""
    result = _write_code(tmp_path, PARENT, parent=PARENT, dry_run=True)

    assert result.passed is True, result.error


# --- gate two: the measurement ----------------------------------------------


def test_identical_metrics_are_not_evidence() -> None:
    """The rogii shape: different executions, same numbers."""
    metrics = {"cv_rmse": 1789.6796883967336, "n_features": 31}

    assert compared_against_itself("E-246", "E-244", metrics, dict(metrics))


def test_a_control_that_is_the_treatment_is_caught() -> None:
    """Control resolution walks several fallbacks and can land on this run."""
    assert compared_against_itself("E-244", "E-244", {"cv_rmse": 1.0}, {"cv_rmse": 2.0})


@pytest.mark.parametrize(
    ("treatment", "control"),
    [
        ({"cv_rmse": 1789.68}, {"cv_rmse": 194.3}),
        # Differing anywhere is enough — the run varied.
        ({"cv_rmse": 1.0, "n_features": 31}, {"cv_rmse": 1.0, "n_features": 30}),
    ],
)
def test_a_real_comparison_is_left_alone(treatment: dict, control: dict) -> None:
    assert compared_against_itself("E-246", "E-244", treatment, control) is None


def test_an_unmeasured_treatment_is_not_this_gates_business() -> None:
    """Empty metrics mean the run produced nothing, which `run_training` owns.
    Claiming it here would report the wrong reason for the wrong failure."""
    assert compared_against_itself("E-246", "E-244", {}, {}) is None


# --- and the wiring, which is the part that protects a belief ---------------


def _compare_with_parent(tmp_path: Path, monkeypatch, parent_metrics: dict) -> list[str]:
    """Run COMPARE against `parent_metrics`; return the applies that happened."""
    from test_engineer_capabilities import _ctx

    from labpilot.research_engine.evidence import apply as apply_mod
    from labpilot.research_engine.evidence import compare_service

    context = _ctx(tmp_path / "knowledge", task_type=TaskType.COMPARE)
    context.plan.metadata["parent_execution_id"] = "E-244"
    context.plan.metadata["parent_metrics"] = parent_metrics
    (context.workspace_root / "metrics.json").write_text(
        json.dumps({"cv_rmse": 1789.6796883967336}), encoding="utf-8"
    )
    # Without a direction the builder refuses to sign a conclusion, which is a
    # different guard and would mask this one.
    (context.workspace_root / "competition.json").write_text(
        json.dumps({"metric": {"key": "rmse", "direction": "minimize"}}), encoding="utf-8"
    )

    applied: list[str] = []
    monkeypatch.setattr(
        apply_mod, "apply_card_to_beliefs", lambda **_: applied.append("beliefs")
    )
    monkeypatch.setattr(
        apply_mod, "apply_card_to_hypothesis", lambda **_: applied.append("hypothesis")
    )
    compare_service.run_compare_and_build_card(context)
    return applied


def test_a_self_comparison_moves_no_belief(tmp_path: Path, monkeypatch) -> None:
    """The failure in full: identical numbers reaching `apply_card_to_hypothesis`
    is what turns a measurement of the control into a verdict on the treatment."""
    applied = _compare_with_parent(
        tmp_path, monkeypatch, {"cv_rmse": 1789.6796883967336}
    )

    assert applied == []


def test_a_real_comparison_still_applies(tmp_path: Path, monkeypatch) -> None:
    """The guard must not cost the evidence loop its whole purpose."""
    applied = _compare_with_parent(tmp_path, monkeypatch, {"cv_rmse": 194.3})

    assert applied == ["beliefs", "hypothesis"]


# --- the gate must judge the proposal, not one file in it -------------------


def _write_code_proposing(tmp_path: Path, files: dict[str, str], *, parent: dict[str, str]):
    """Run WRITE_CODE with a proposal over `files`, atop an existing `parent` tree."""
    import json as _json

    from helpers.baseline_campaign import CodeFileSpec, CodeProposal
    from test_engineer_capabilities import _ctx

    from labpilot.research_engine.execution.capabilities.code_engineering import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = _ctx(tmp_path / "knowledge", task_type=TaskType.WRITE_CODE, constraints={})
    (context.workspace_root / "profile.json").write_text(
        _json.dumps({"competition": "demo", "target_column": "target", "id_column": "id"}),
        encoding="utf-8",
    )
    pipeline = context.workspace_root / "pipeline"
    pipeline.mkdir(parents=True, exist_ok=True)
    for name, body in parent.items():
        (context.workspace_root / name).write_text(body, encoding="utf-8")

    class _Agent:
        last_used_llm = True

        def run(self, ctx):
            return CodeProposal(
                summary="artifact under test",
                files=[
                    CodeFileSpec(path=p, content=c, action="write") for p, c in files.items()
                ],
            )

    capability = CodeEngineeringCapability()
    capability._agent = _Agent()  # noqa: SLF001 — the injection point
    return capability.execute(context), context.workspace_root


HELPER = "def clip(x):\n    return x\n"


def test_a_delta_confined_to_another_pipeline_file_is_a_real_experiment(tmp_path: Path) -> None:
    """The false reject. `pipeline/infer.py` is a first-class edit target.

    `_copy_tree` offers aider every `*.py` under `pipeline/`, the platform
    creates `infer.py` itself when it is absent, and the coding prompt names it
    — so a hypothesis about post-processing produces a proposal with no
    `train.py` in it at all. Judged by train.py alone that read as "the proposal
    left the pipeline unchanged", with the edit already written to disk.
    """
    result, root = _write_code_proposing(
        tmp_path,
        {"pipeline/infer.py": HELPER.replace("return x", "return x.clip(0, 1)")},
        parent={"pipeline/train.py": PARENT, "pipeline/infer.py": HELPER},
    )

    assert result.passed is True, result.error
    assert (root / "pipeline" / "infer.py").read_text() != HELPER


def test_a_proposal_that_rewrites_every_file_identically_is_still_refused(tmp_path: Path) -> None:
    """The gate widened, not weakened: no file moved, so nothing was tested."""
    result, _ = _write_code_proposing(
        tmp_path,
        {"pipeline/train.py": PARENT, "pipeline/infer.py": HELPER},
        parent={"pipeline/train.py": PARENT, "pipeline/infer.py": HELPER},
    )

    assert result.passed is False
    assert "differs_from_parent" in result.checks


def test_one_moved_file_among_several_is_enough(tmp_path: Path) -> None:
    """A treatment differs from its control if *anything* differs."""
    result, _ = _write_code_proposing(
        tmp_path,
        {"pipeline/train.py": PARENT, "pipeline/infer.py": HELPER + "# tuned\n"},
        parent={"pipeline/train.py": PARENT, "pipeline/infer.py": HELPER},
    )

    assert result.passed is True, result.error


# --- the disqualification has to travel on the card -------------------------
#
# Skipping `apply_card_to_*` was never enough: the card is persisted and
# `comparison.json` written *before* that point, so reflection read the refused
# numbers straight back and confirmed the hypothesis, and `submit_learn`
# re-derived a decision from the same card without consulting the check.


def _card(tmp_path: Path, *, treatment: str, control: str | None, metrics, control_metrics):
    from labpilot.research_engine.evidence.builder import build_evidence_card

    (tmp_path / "competition.json").write_text(
        json.dumps({"metric": {"key": "rmse", "direction": "minimize"}}), encoding="utf-8"
    )
    return build_evidence_card(
        knowledge_dir=tmp_path / "knowledge",
        competition="demo",
        treatment_execution_id=treatment,
        treatment_metrics=metrics,
        hypothesis_id="H-021",
        control_execution_id=control,
        control_metrics=control_metrics,
        workspace_root=tmp_path,
        persist=False,
    )


def test_a_self_comparison_is_inconclusive_on_the_card_itself(tmp_path: Path) -> None:
    """Not a log line in one caller — a field every reader sees."""
    card = _card(
        tmp_path,
        treatment="E-244",
        control="E-244",
        metrics={"cv_rmse": 1.0},
        control_metrics={"cv_rmse": 2.0},
    )

    assert card.decision.value == "inconclusive"
    assert card.uncomparable_reason
    assert "same execution" in card.uncomparable_reason


def test_identical_metrics_are_inconclusive_on_the_card_itself(tmp_path: Path) -> None:
    metrics = {"cv_rmse": 1789.6796883967336, "n_features": 31}
    card = _card(
        tmp_path,
        treatment="E-246",
        control="E-244",
        metrics=metrics,
        control_metrics=dict(metrics),
    )

    assert card.decision.value == "inconclusive"
    assert "behaviourally inert" in (card.uncomparable_reason or "")


def test_a_real_comparison_still_signs_a_verdict(tmp_path: Path) -> None:
    """The guard must not cost the evidence loop its purpose."""
    card = _card(
        tmp_path,
        treatment="E-246",
        control="E-244",
        metrics={"cv_rmse": 100.0},
        control_metrics={"cv_rmse": 200.0},
    )

    assert card.uncomparable_reason is None
    assert card.decision.value == "accepted"


def test_a_leaderboard_gain_cannot_rescue_a_self_comparison(tmp_path: Path) -> None:
    """`submit_learn`'s re-derivation, which is how the gate was bypassed.

    A self-comparison has a control id and a `parent_cv`, so it cleared that
    function's `missing_control` test; with `cv_gain` None and a non-negative
    leaderboard delta, `_decide` returned `accepted` and the hypothesis was
    confirmed on a measurement of the control.
    """
    from labpilot.research_engine.evidence.builder import decide_evidence

    card = _card(
        tmp_path,
        treatment="E-244",
        control="E-244",
        metrics={"cv_rmse": 1.0},
        control_metrics={"cv_rmse": 2.0},
    )

    decision, _ = decide_evidence(
        cv_gain=card.observed.cv_gain,
        lb_gain=0.5,
        stability=card.observed.stability,
        maximize=card.maximize,
        missing_control=(card.control_experiment is None and card.observed.parent_cv is None)
        or card.uncomparable_reason is not None,
    )

    assert decision.value == "inconclusive"


def test_the_written_comparison_says_it_is_not_evidence(tmp_path: Path, monkeypatch) -> None:
    """`comparison.json` is what reflection reads, so it has to carry the verdict."""
    from test_engineer_capabilities import _ctx

    from labpilot.research_engine.evidence import compare_service
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    context = _ctx(tmp_path / "knowledge", task_type=TaskType.COMPARE)
    context.plan.metadata["parent_execution_id"] = "E-244"
    context.plan.metadata["parent_metrics"] = {"cv_rmse": 1789.6796883967336}
    (context.workspace_root / "metrics.json").write_text(
        json.dumps({"cv_rmse": 1789.6796883967336}), encoding="utf-8"
    )
    (context.workspace_root / "competition.json").write_text(
        json.dumps({"metric": {"key": "rmse", "direction": "minimize"}}), encoding="utf-8"
    )

    compare_service.run_compare_and_build_card(context)

    written = json.loads((context.workspace_root / "comparison.json").read_text())
    assert written["decision"] == "inconclusive"
