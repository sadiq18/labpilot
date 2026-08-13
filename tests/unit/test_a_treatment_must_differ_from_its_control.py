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
