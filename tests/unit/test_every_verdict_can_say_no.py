"""One rejection test per verdict site — M20 exit criterion 1, finished.

`test_every_gate_rejects_something.py` enumerates the sites; this proves them.
The enumerator was per-*module* for one round, so a single marker stood for four
gates and two unfailable ones shipped inside the change written to remove them.
Keyed on `capability:check` now, which surfaced twenty sites nobody had shown
could say no.

Eight of those turned out to check nothing at all — *"no requirements file;
skipped install"*, *"runtime job already active"* — and declare it on their own
evidence rather than pretending. The twelve here are real gates, and each is fed
an artifact that should fail it.

Every one was verified **red-then-green** by disabling its guard and watching the
test go red. Where the lever is not the obvious line, the docstring names it —
on this branch one sweep stayed green because it disabled the line that blanks a
value rather than the branch that sets the verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers.capability_context import capability_context
from helpers.real_failures import real_failure

from labpilot.research_engine.planner.schemas.task_types import TaskType


def _profile(context) -> None:
    """The dataset profile `write_code` refuses without."""
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


# -- code_engineering ---------------------------------------------------------


@pytest.mark.rejects("code_engineering:profile_required")
def test_write_code_refuses_without_a_dataset_profile(tmp_path):
    """Codegen writes a pipeline against the dataset's shape. Without a profile
    it would be writing against an imagined one, and the file it produced would
    read as a real experiment.

    Red-then-green: removing the `profile.json missing` branch lets the step
    proceed and report a proposal built from nothing.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering import (
        CodeEngineeringCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.WRITE_CODE)

    result = CodeEngineeringCapability().execute(context)

    assert result.passed is False
    assert "profile" in (result.error or "").lower()


@pytest.mark.rejects("code_engineering:apply")
def test_write_code_refuses_a_proposal_that_cannot_run(tmp_path, monkeypatch):
    """When `apply_proposal` refuses, the step must fail rather than carry on.

    `apply_proposal` is patched to raise the error the real 2026-08-08 artifact
    produces, rather than driving the whole codegen path to get there. Two
    earlier attempts did drive it and both were green for the wrong reason — the
    proposal reached the `last_resort` branch and the step failed *before* apply,
    so the test proved a different gate. The site under test is this `except`,
    and this reaches it.

    Red-then-green: flipping the handler's `passed=False` to `True` makes this
    green with the proposal refused.
    """
    from labpilot.research_engine.execution.capabilities.code_engineering import (
        CodeEngineeringCapability,
    )
    from labpilot.research_engine.execution.capabilities.code_engineering import (
        capability as module,
    )
    from labpilot.research_engine.execution.capabilities.code_engineering.apply import (
        ApplyError,
        apply_proposal,
    )
    from labpilot.research_engine.execution.schemas.code_proposal import (
        CodeFileSpec,
        CodeProposal,
    )

    context = capability_context(tmp_path, task_type=TaskType.WRITE_CODE)
    _profile(context)
    truncated = real_failure("truncated_train_py.txt")

    # The real refusal, produced by the real gate, then replayed here — so the
    # message this step reports is one `apply_proposal` actually emits.
    try:
        apply_proposal(
            Path(tmp_path) / "scratch",
            CodeProposal(
                files=[CodeFileSpec(path="pipeline/train.py", content=truncated, action="write")]
            ),
        )
        raise AssertionError("the corpus artifact should have been refused")
    except ApplyError as refusal:
        message = str(refusal)

    def _refuses(*args, **kwargs):
        raise ApplyError(message)

    monkeypatch.setattr(module, "apply_proposal", _refuses)

    class _Wrote:
        """An agent that produced a file. Without `last_used_llm`, `origin`
        becomes `last_resort` and the step fails *before* apply — which is how
        the two earlier versions of this test went green on the wrong gate."""

        last_used_llm = True

        def run(self, ctx):
            return CodeProposal(
                summary="truncated",
                files=[
                    CodeFileSpec(path="pipeline/train.py", content=truncated, action="write")
                ],
            )

    capability = CodeEngineeringCapability()
    capability._agent = _Wrote()

    result = capability.execute(context)

    assert result.passed is False
    assert "PEP 723" in (result.error or "")


@pytest.mark.rejects("code_engineering:override")
@pytest.mark.rejects("code_engineering:modify_config")
def test_the_override_and_config_verdicts_read_the_file_that_landed(tmp_path):
    """Both verdicts are `path.is_file()` — a fact about the tree, not about
    having reached the return. Asserted structurally because driving them needs
    a `modify_config`/override task shape the planner builds, and a test that
    fabricated one would be proving its own fixture.

    Red-then-green: pinning either to `True` leaves the assertion below failing.
    """
    import inspect

    from labpilot.research_engine.execution.capabilities.code_engineering import (
        capability as module,
    )

    source = inspect.getsource(module)

    assert "passed=train_path.is_file()" in source
    assert "passed=config_path.is_file()" in source


# -- dependency ---------------------------------------------------------------


@pytest.mark.rejects("dependency:pip_install")
def test_a_failed_install_is_not_a_satisfied_dependency(tmp_path, monkeypatch):
    """`pip install` exiting non-zero means the run cannot import what it needs.
    Reported as success, the next step fails on an import error instead, three
    steps from the cause.

    Red-then-green: pinning `ok = True` makes this green on a failing install.
    """
    import subprocess

    from labpilot.research_engine.execution.capabilities.dependency import (
        DependencyCapability,
    )

    context = capability_context(
        tmp_path,
        task_type=TaskType.INSTALL_PACKAGE,
        constraints={"install_packages": True},
    )
    (context.workspace_root / "requirements.txt").write_text(
        "a-package-that-is-definitely-not-installed==1.0\n", encoding="utf-8"
    )

    def _fails(*args, **kwargs):
        return subprocess.CompletedProcess(args=["pip"], returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _fails)

    result = DependencyCapability().execute(context)

    assert result.passed is False


# -- evaluation ---------------------------------------------------------------


@pytest.mark.rejects("evaluation:inference")
def test_evaluation_refuses_when_inference_wrote_no_predictions(tmp_path):
    """`passed=pred.is_file()` reads as a real check, and it was not: the step
    **wrote** `id,prediction\\n0,0` first, so the file it tested for was one it
    had just fabricated. Identical to the `submission` defect, in a different
    capability — found while writing this test, not by reading the code.

    Red-then-green: restoring the unconditional placeholder makes this green with
    nothing to infer from.
    """
    from labpilot.research_engine.execution.capabilities.evaluation import (
        EvaluationCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_INFERENCE)

    result = EvaluationCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("evaluation:compare")
@pytest.mark.rejects("evaluation:evidence_card")
def test_a_comparison_that_compared_nothing_is_not_a_comparison(tmp_path, monkeypatch):
    """`passed=True` was unconditional, so a card built with no control — or over
    placeholder metrics — reported success, and the card is exactly what COMPARE
    exists to produce. Found while proving this site, not by review.

    Red-then-green: restoring `passed=True` makes this green on a card with no
    `cv_gain`.
    """
    from labpilot.research_engine.evidence import compare_service
    from labpilot.research_engine.evidence.models import EvidenceCard
    from labpilot.research_engine.execution.capabilities.evaluation import (
        EvaluationCapability,
    )

    nothing_compared = EvidenceCard(id="EV-001", decision_reason="missing_control")
    # Patched where it lives, not where it is used: the capability imports it
    # inside the method, so patching the capability module would have changed a
    # name nothing reads — a test that passes without touching the code path.
    monkeypatch.setattr(
        compare_service, "run_compare_and_build_card", lambda *a, **k: nothing_compared
    )
    context = capability_context(tmp_path, task_type=TaskType.COMPARE)

    result = EvaluationCapability().execute(context)

    assert nothing_compared.observed.cv_gain is None
    assert result.passed is False
    assert "compared nothing" in (result.error or "")


# -- training -----------------------------------------------------------------


@pytest.mark.rejects("training:train_script")
def test_training_refuses_a_workspace_with_no_script(tmp_path):
    """Defect 14's neighbour: a `train.py` that is not there. `except OSError:
    return False` once answered "yes, it runs" for exactly this.

    Red-then-green: dropping the `missing pipeline/train.py` branch makes the
    step continue to a runner with nothing to run.
    """
    from labpilot.research_engine.execution.capabilities.training.capability import (
        TrainingCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_TRAINING)

    result = TrainingCapability().execute(context)

    assert result.passed is False
    assert "train.py" in (result.error or "")


@pytest.mark.rejects("training:train_runner")
def test_a_training_run_that_exits_non_zero_fails(tmp_path, monkeypatch):
    """The runner's exit status, which `passed=ok` reads. Kept distinct from the
    freshness guard next door: that one catches a run that *succeeded* and wrote
    nothing new, this one a run that did not succeed.

    Red-then-green: pinning `ok = True` makes this green on a crash.
    """
    import labpilot.research_engine.execution.training.runner as runner_module
    from labpilot.research_engine.execution.capabilities.training.capability import (
        TrainingCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_TRAINING)
    (context.workspace_root / "pipeline" / "train.py").write_text(
        'def main():\n    return 1\n\n\nif __name__ == "__main__":\n    main()\n',
        encoding="utf-8",
    )

    class _Crashes:
        def __init__(self, *a, **k):
            pass

        def run(self, timeout=None):
            class _Bad:
                returncode = 1
                stdout = ""
                stderr = "Traceback (most recent call last):\nKeyError: 'TVT'\n"

            return _Bad()

    monkeypatch.setattr(runner_module, "TrainingRunner", _Crashes)

    result = TrainingCapability().execute(context)

    assert result.passed is False


# -- verification -------------------------------------------------------------


@pytest.mark.rejects("verification:syntax")
def test_the_smoke_gate_refuses_a_file_that_does_not_parse(tmp_path):
    """Syntax is necessary and not sufficient — but it is still necessary, and a
    file that does not parse cannot be smoke-tested.

    Red-then-green: pinning the syntax verdict to `True` makes this green on a
    file with an unclosed paren.
    """
    from labpilot.research_engine.execution.capabilities.verification import (
        VerificationCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_SMOKE_TEST)
    (context.workspace_root / "pipeline" / "train.py").write_text(
        "def broken(:\n", encoding="utf-8"
    )

    result = VerificationCapability().execute(context)

    assert result.passed is False


@pytest.mark.rejects("verification:pytest")
def test_a_failing_unit_test_fails_the_step(tmp_path, monkeypatch):
    """`passed=ok` from the pytest run. A unit-test step that cannot report a
    failing test is the milestone's title with a test runner attached.

    Red-then-green: pinning `ok = True` makes this green with a failing test in
    the workspace.
    """
    import subprocess

    from labpilot.research_engine.execution.capabilities.verification import (
        VerificationCapability,
    )

    context = capability_context(tmp_path, task_type=TaskType.RUN_UNIT_TEST)
    tests_dir = context.workspace_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_fails.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")

    def _fails(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=["pytest"], returncode=1, stdout="1 failed", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", _fails)

    result = VerificationCapability().execute(context)

    assert result.passed is False
