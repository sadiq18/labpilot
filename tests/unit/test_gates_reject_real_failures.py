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

**Three of the first nine tests here proved nothing**, and the sweep is what
found them — not review, which had passed all three:

* `code_engineering` refused at an earlier precondition (*"missing dataset
  profile"*), never reaching the branch it claimed to test;
* `training` had a second, weaker copy of a test that already existed properly
  elsewhere — the marker moved to the real one and the copy went;
* `research_review` drove `force_block`, which is a test hook, so the proof was
  nearly circular — and with the hook disabled the capability failed anyway, for
  an unrelated reason.

That is the milestone's own claim landing on itself: each read as correct, and
each said pass. The lever matters too — `training`'s first attempt disabled the
line that blanks the metrics rather than the branch that sets the verdict, and
stayed green.
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
    metrics evaluate would later dress up as a leaderboard result.

    **The profile is here because the first version of this test proved
    nothing.** Without it the capability refused at an earlier precondition —
    *"missing dataset profile"* — so the assertion held with the last-resort
    guard disabled, and the test was green either way. Found by the
    red-then-green sweep this milestone requires, in a test written to enforce
    it.

    Red-then-green, verified 2026-08-09: disabling the
    `origin == "last_resort" and not is_dry_run` branch makes this pass.
    """
    context = capability_context(tmp_path, task_type=TaskType.WRITE_CODE)
    (context.workspace_root / "profile.json").write_text(
        json.dumps(
            {
                "competition": "demo",
                "files": ["train.csv"],
                "train_file": "train.csv",
                "test_file": "test.csv",
                "sample_submission_file": "sample_submission.csv",
                "target_column": "y",
                "id_column": "id",
                "columns": [{"name": "id", "dtype": "int"}, {"name": "y", "dtype": "float"}],
                "row_count": 10,
            }
        ),
        encoding="utf-8",
    )

    result = CodeEngineeringCapability().execute(context)

    assert result.passed is False
    assert "no files" in (result.error or "").lower() or "codegen" in (result.error or "").lower()


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
def test_research_review_rejects_a_script_with_no_entry_point(tmp_path):
    """The real 2026-08-08 artifact: 624 bytes of docstring and half a comment,
    which `ast.parse` accepted and `run_smoke_test` passed because a file that
    does nothing exits 0.

    **The first version of this test drove `force_block` and proved nothing.**
    That is a test hook, so proving the gate through it is close to circular —
    and with the hook disabled the capability failed anyway, on the real review
    finding a missing `train.py`. Found by the red-then-green sweep.

    Red-then-green, verified 2026-08-09: neutering `_has_standard_main_guard`
    makes this pass.
    """
    context = capability_context(tmp_path, task_type=TaskType.RESEARCH_REVIEW)
    (context.workspace_root / "pipeline" / "train.py").write_text(
        real_failure("truncated_train_py.txt"), encoding="utf-8"
    )

    result = ResearchReviewCapability().execute(context)

    assert result.passed is False
    assert "__main__" in (result.error or "") or "entrypoint" in (result.error or "")


@pytest.mark.rejects("submission")
def test_submission_refuses_when_nothing_was_packaged(tmp_path):
    """`passed=packaged.is_file()` read as a real check, and it is the eighth
    instance of the shape: the capability *writes* `submission_E-001.csv`
    itself, so the file it tests for is one it just created. The verdict asks
    "did I write a file" while promising "a submission was built" — and it says
    pass on a workspace with no model, no predictions and no data.

    Fixed 2026-08-09: a real run with nothing to submit now refuses instead of
    fabricating `id,prediction\n0,0`. The placeholder survives only under
    `--dry-run`, where the wiring *is* what is being checked.
    """
    from labpilot.research_engine.execution.capabilities.submission import (
        SubmissionCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.BUILD_SUBMISSION)

    result = SubmissionCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("workspace")
def test_workspace_refuses_a_tree_it_could_not_prepare(tmp_path):
    """`passed=passed` looked computed, and it is — from whether the *directories*
    exist. Run against a workspace with no Kaggle credentials and no data, the
    step reports `passed=True` with `download_skipped: no_kaggle_config` and
    `profile_skipped: no_data` in its own metadata: it says so, and passes
    anyway. Every step downstream then runs against an empty tree, which is how
    a campaign spends nine runs discovering there was never any data.

    Fixed 2026-08-09 by separating *skipped because asked to* from *skipped
    because unable*. Both were `None`, and the verdict read anything-but-False
    as done.
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


@pytest.mark.rejects("runtime")
def test_runtime_refuses_to_substitute_for_a_runtime_it_cannot_resolve(tmp_path):
    """A runtime that was *asked for* and could not be found is the one thing
    this step can get wrong, and it was the one thing it could not report: the
    lookup fell through to the local default and the card read "selected runtime
    local". A campaign that asked for a GPU trained somewhere else and called it
    a success.

    Red-then-green: restoring the bare `except Exception: runtime_id =
    self._default` makes this pass with a fabricated runtime name.
    """
    from labpilot.research_engine.execution.capabilities.runtime import RuntimeCapability

    context = capability_context(
        tmp_path,
        task_type=TaskType.SELECT_RUNTIME,
        constraints={"runtime_id": "a100-cluster-that-does-not-exist"},
    )

    result = RuntimeCapability().execute(context)

    assert result.passed is False
    assert "a100-cluster-that-does-not-exist" in (result.error or "")


@pytest.mark.rejects("reporting")
def test_reporting_refuses_to_report_on_a_run_that_produced_nothing(tmp_path):
    """Four return sites, every one `passed=True`, because each ends by writing
    a file and `write_text` either works or raises. The verdict answered "did I
    write something" while the step promises "this execution was reported on" —
    so a run with no metrics still got a report, and the card read clean.

    Red-then-green: pinning `passed=True` makes this green against an empty
    workspace.
    """
    from labpilot.research_engine.execution.capabilities.reporting import (
        ReportingCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.GENERATE_REPORT)

    result = ReportingCapability().execute(context)

    assert result.passed is False
    assert "metrics" in (result.error or "")
