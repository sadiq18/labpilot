"""M20 exit criterion 1: each gate, shown the artifact it must refuse.

One test per capability that reports a verdict, each marked
`@pytest.mark.rejects("<name>")` so `test_every_gate_rejects_something.py` can
tell that the claim was made. The marker is the claim; the assertion is the
proof.

**Red-then-green, by hand.** A test that passes with and without the fix has
proven nothing, and that cannot be automated — so each docstring below says what
was broken to confirm the test fails. Where the guard predates this file, the
method was to stash the guard's condition, watch the test go red, and restore.

Where a real 2026-08-08 artifact exists it is used, per
`15-gates-must-fail.md`'s trap: *"do not test the guard against a synthetic bad
input when a real one exists."*
"""

from __future__ import annotations

import json

import pytest
from helpers.capability_context import capability_context
from helpers.real_failures import real_failure

from labpilot.research_engine.execution.capabilities.code_engineering import (
    CodeEngineeringCapability,
)
from labpilot.research_engine.execution.capabilities.evaluation import EvaluationCapability
from labpilot.research_engine.execution.capabilities.research_review import (
    ResearchReviewCapability,
)
from labpilot.research_engine.execution.capabilities.verification import VerificationCapability
from labpilot.research_engine.planner.schemas.task_types import TaskType


@pytest.mark.rejects("code_engineering")
def test_code_engineering_refuses_when_it_produced_no_code(tmp_path):
    """The 2026-08-08 shape, at the capability boundary: codegen produces
    nothing, and the step must fail rather than continue on a stub whose fake
    metrics evaluate would dress up as a leaderboard result.

    Red-then-green: with the `origin == "last_resort"` guard removed, this
    returns `passed=True` and writes the stub.
    """
    context = capability_context(tmp_path, task_type=TaskType.WRITE_CODE)

    result = CodeEngineeringCapability().execute(context)

    assert result.passed is False
    assert "codegen" in (result.error or "").lower() or "code" in (result.summary or "").lower()


@pytest.mark.rejects("evaluation")
def test_evaluation_refuses_a_run_that_wrote_no_metrics(tmp_path):
    """Defect 5's neighbour: a run that produced nothing must not evaluate. The
    capability reads `metrics.json`, and its absence is the honest answer that
    a crashed run has no score — not the previous run's.

    Red-then-green: with the `metrics.json missing` branch returning
    `passed=True`, this goes green on an empty workspace.
    """
    context = capability_context(tmp_path, task_type=TaskType.EVALUATE)

    result = EvaluationCapability().execute(context)

    assert result.passed is False
    assert "metrics" in (result.error or "")


@pytest.mark.rejects("verification")
def test_verification_refuses_a_workspace_with_no_training_script(tmp_path):
    """Defect 14 by another door: a `train.py` that is not there used to answer
    "yes, it runs" — `except OSError: return False` on the unrunnable check.
    The capability's own answer must be that there is nothing to verify.

    Red-then-green: with the `not train.is_file()` branch removed, this returns
    a pass for an empty `pipeline/`.
    """
    context = capability_context(tmp_path, task_type=TaskType.RUN_SMOKE_TEST)

    result = VerificationCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("research_review")
def test_research_review_refuses_when_the_plan_says_to_block(tmp_path):
    """The review gate exists to stop a plan, and a gate that cannot stop one is
    the milestone's title. Driven through the same metadata the planner writes.

    Red-then-green: with the `force_block` branch removed, this passes.
    """
    context = capability_context(
        tmp_path,
        task_type=TaskType.RESEARCH_REVIEW,
        metadata={"force_block": True},
    )

    result = ResearchReviewCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("training")
def test_training_refuses_a_stale_metrics_file(tmp_path):
    """Defect 5, exactly: E-147 died on `import catboost` and reported
    `rmse 13.957107` — E-003's figure from six days earlier, still on disk. The
    freshness guard is what separates "this run scored" from "a file exists".

    Red-then-green: with the `mtime >= wall_started` comparison removed, the
    stale figure is published as this run's result.
    """
    import os
    import time

    from labpilot.research_engine.execution.capabilities.training.capability import (
        TrainingCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_TRAINING)
    stale = context.workspace_root / "metrics.json"
    stale.write_text(json.dumps({"cv_rmse": 13.957107}), encoding="utf-8")
    old = time.time() - 86_400
    os.utime(stale, (old, old))

    result = TrainingCapability().execute(context)

    assert result.passed is False
    assert result.metadata.get("metrics") in (None, {}, {"metrics": {}})


@pytest.mark.rejects("submission")
@pytest.mark.xfail(strict=True, reason="M20 finding 2026-08-09: verdict is weaker than the promise")
def test_submission_refuses_when_nothing_was_packaged(tmp_path):
    """**Currently fails, and that is the finding.**

    `passed=packaged.is_file()` reads as a real check, and it is the eighth
    instance of the shape: the capability *writes* `submission_E-001.csv`
    itself, so the file it tests for is one it just created. The verdict asks
    "did I write a file" while promising "a submission was built" — and it says
    pass on a workspace with no model, no predictions and no data.

    Strict xfail rather than a `_KNOWN` list: when the verdict starts meaning
    what it says, this test fails until someone deletes the marker. A list
    needs a reader; this does not.
    """
    from labpilot.research_engine.execution.capabilities.submission import (
        SubmissionCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.BUILD_SUBMISSION)

    result = SubmissionCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("workspace")
@pytest.mark.xfail(strict=True, reason="M20 finding 2026-08-09: skipped work counts as done")
def test_workspace_refuses_a_tree_it_could_not_prepare(tmp_path):
    """**Currently fails, and that is the finding.**

    `passed=passed` looks computed, and it is — from whether the *directories*
    exist. Run against a workspace with no Kaggle credentials and no data, the
    step reports `passed=True` with `download_skipped: no_kaggle_config` and
    `profile_skipped: no_data` in its own metadata: it says so, and passes
    anyway. Every step downstream then runs against an empty tree, which is how
    a campaign spends nine runs discovering there was never any data.

    Strict xfail, per the note on the submission case above.
    """
    from labpilot.research_engine.execution.capabilities.workspace import (
        WorkspaceCapability,
    )

    context = capability_context(
        tmp_path,
        task_type=TaskType.PREPARE_WORKSPACE,
        constraints={"skip_download": False, "dry_run": False},
    )

    result = WorkspaceCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("dependency")
def test_dependency_refuses_a_stdlib_module_in_the_block(tmp_path):
    """Defect 11, from the real artifact: codegen declared `glob`, and uv
    refused all six dependencies — the run never started, so every gate
    downstream saw nothing at all.

    Red-then-green: without `sys.stdlib_module_names` filtering, `glob` survives
    into the resolved set.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
        strip_stdlib_dependencies,
    )

    _, dropped = strip_stdlib_dependencies(real_failure("stdlib_dependency_block.txt"))

    assert dropped == ["glob"]
