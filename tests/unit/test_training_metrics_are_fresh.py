"""Training must write metrics, not merely find some.

Measured on rogii 2026-08-09: execution E-227 reported **succeeded** and plan
P-025 went **done** against a `metrics.json` written the previous evening. A
green plan with no result, and the number on the card belonged to a different
experiment.

`run_experiment` closed this hole with `_metrics_written_since`. The Engineer
path — `research run`, which is what a plan actually goes through — never had
the guard.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


class _Result:
    returncode = 0
    stdout = ""
    stderr = ""


class _Runner:
    """A training run that exits 0 and writes nothing."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, timeout=None):
        return _Result()

    def save_run_log(self, result):
        path = self.root / "artifacts" / "run.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return path

    def collect_artifacts(self):
        return {}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "train.py").write_text("print('x')\n", encoding="utf-8")
    monkeypatch.setattr(
        "labpilot.research_engine.execution.training.runner.TrainingRunner", _Runner
    )
    return tmp_path


def _run(workspace):
    from labpilot.research_engine.execution.capabilities.training.capability import (
        TrainingCapability,
    )

    class _Ctx:
        workspace_root = workspace
        constraints: dict = {}

        class task:
            id = "T-1"
            metadata: dict = {}

        class plan:
            id = "P-1"

        class execution:
            id = "E-1"

    return TrainingCapability().execute(_Ctx())


@pytest.mark.rejects("training:execute()")
def test_a_stale_metrics_file_does_not_count_as_a_result(workspace):
    """The exact E-227 shape: yesterday's metrics, today's green plan.

    Carries M20's `rejects` marker because it is this capability's proof, and it
    was already built the way the milestone asks: a runner double so the run
    genuinely completes, leaving the freshness guard as the only thing standing
    between a stale figure and a published result.

    Red-then-green, verified 2026-08-09 by disabling `if ok and not fresh:` —
    the stale `194.8` is then published as this run's score. Worth recording
    that the *first* lever tried was `metrics if fresh else {}`, which left the
    test green: that line blanks the figure, the verdict lives one branch up,
    and picking the wrong lever proves nothing just as surely as a weak test
    does.
    """
    stale = workspace / "metrics.json"
    stale.write_text(json.dumps({"cv_rmse": 194.8}), encoding="utf-8")
    old = time.time() - 86_400
    import os

    os.utime(stale, (old, old))

    result = _run(workspace)

    assert result.passed is False
    assert "predates this run" in (result.error or "")


def test_no_metrics_at_all_still_fails_with_its_own_message(workspace):
    """The two causes need different messages — a missing file points at the
    __main__ guard, a stale one at the previous execution."""
    result = _run(workspace)

    assert result.passed is False
    assert "__main__" in (result.error or "")


def test_metrics_written_by_this_run_pass(workspace):
    """The carve-out must not cost the behaviour it guards."""

    class _WritingRunner(_Runner):
        def run(self, timeout=None):
            (self.root / "metrics.json").write_text(json.dumps({"cv_rmse": 1.0}), encoding="utf-8")
            return _Result()

    import labpilot.research_engine.execution.training.runner as runner_mod

    runner_mod.TrainingRunner = _WritingRunner

    result = _run(workspace)

    assert result.passed is True, result.error
    assert result.metrics["cv_rmse"] == 1.0


def test_the_real_e227_file_would_now_be_rejected(workspace):
    """The actual artifact that made P-025 green, reproduced from its recorded
    contents and mtime. No training run — the guard is a filesystem question.

    `cv_rmse: 194.80084243002463` is the parent H-003 number, which is how the
    card came to carry a result from a different experiment.
    """
    import os

    stale = workspace / "metrics.json"
    stale.write_text(
        json.dumps(
            {
                "cv_rmse": 194.80084243002463,
                "mse": 194.80084243002463,
                "rmse": 13.957107237175784,
                "n_features": 20,
            }
        ),
        encoding="utf-8",
    )
    # 2026-08-08 20:35 against a run starting 2026-08-09 07:52.
    written_yesterday = time.time() - (11 * 3600 + 17 * 60)
    os.utime(stale, (written_yesterday, written_yesterday))

    result = _run(workspace)

    assert result.passed is False
    assert "earlier execution" in (result.error or "")


def test_a_silent_training_failure_makes_the_code_suspect():
    """The narrow case, and only it.

    `RUN_TRAINING` is deliberately not a code-validation task — training fails
    for reasons code cannot fix, and rebuilding would discard a file that
    passed its gates. But a run that *exits 0* and writes nothing did not
    crash; it failed to do its job. On rogii 2026-08-09 that was a script
    writing to `./workspace/metrics.json`, a directory it invented, and three
    retries rebuilt blind because nothing marked the code suspect.
    """
    from labpilot.research_engine.execution.capabilities.training.capability import (
        METRICS_NOT_WRITTEN,
    )
    from labpilot.research_engine.execution.engineer import ResearchEngineer
    from labpilot.research_engine.planner.schemas.task_types import TaskStatus, TaskType

    class _Task:
        def __init__(self, error):
            self.type = TaskType.RUN_TRAINING
            self.status = TaskStatus.FAILED
            self.metadata = {"error": error}

    class _Plan:
        def __init__(self, error):
            self.tasks = [_Task(error)]

    silent = _Plan(f"training exited 0 but {METRICS_NOT_WRITTEN} \u2014 predates this run")
    crashed = _Plan("CUDA out of memory")

    assert ResearchEngineer._training_produced_nothing(silent)
    assert not ResearchEngineer._training_produced_nothing(crashed)


def test_training_is_still_not_a_code_validation_task():
    """The existing rule stands: a crash or an OOM must not throw away code
    that passed its gates."""
    from labpilot.research_engine.execution.engineer import ResearchEngineer
    from labpilot.research_engine.planner.schemas.task_types import TaskType

    assert TaskType.RUN_TRAINING not in ResearchEngineer._CODE_VALIDATION_TASKS


def test_the_error_names_where_the_output_actually_went(workspace):
    """ "It did not write metrics.json" is true and unhelpful when the script
    wrote one enthusiastically into a directory it invented. rogii burned three
    retries on `./workspace/metrics.json` — each told what was missing, never
    where its output had gone, so each edited something else."""

    class _MisplacedRunner(_Runner):
        def run(self, timeout=None):
            out = self.root / "workspace"
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text(json.dumps({"cv_rmse": 1.0}), encoding="utf-8")
            return _Result()

    import labpilot.research_engine.execution.training.runner as runner_mod

    runner_mod.TrainingRunner = _MisplacedRunner

    result = _run(workspace)

    assert result.passed is False
    assert "workspace/metrics.json" in (result.error or "")
    assert "do not create a directory" in (result.error or "")


def test_an_empty_metrics_file_is_not_a_result(workspace):
    """Reported on PR #117. Freshness answers *when*; this answers whether
    there is a result at all. A run that exits 0 and writes `{}` passes every
    timing check and reports success with nothing measured."""

    class _EmptyRunner(_Runner):
        def run(self, timeout=None):
            (self.root / "metrics.json").write_text("{}", encoding="utf-8")
            return _Result()

    import labpilot.research_engine.execution.training.runner as runner_mod

    runner_mod.TrainingRunner = _EmptyRunner

    result = _run(workspace)

    assert result.passed is False
    assert "no metrics" in (result.error or "")


def test_stale_metrics_do_not_ride_along_with_the_failure(workspace):
    """Reported on PR #118.

    The stale file was still loaded and returned beside `passed=False`, so any
    reader trusting `evidence.metrics` without checking `passed` first saw a
    plausible number belonging to an earlier execution — the same file, and the
    same confusion, the freshness guard exists to end.
    """
    import os

    stale = workspace / "metrics.json"
    stale.write_text(json.dumps({"cv_rmse": 194.8}), encoding="utf-8")
    old = time.time() - 86_400
    os.utime(stale, (old, old))

    result = _run(workspace)

    assert result.passed is False
    assert result.metrics == {}


def test_a_crash_keeps_the_exception_not_the_progress_bar(workspace):
    """The smoke gate was switched to `failure_excerpt` and this path was not,
    so a crash whose stderr opens with a tqdm bar handed the retry the bar."""

    class _CrashRunner(_Runner):
        def run(self, timeout=None):
            class _Bad:
                returncode = 1
                stdout = ""
                stderr = (
                    "\r".join(f"Loading train: {i}%|" for i in range(400)) + "\nKeyError: 'TVT'\n"
                )

            return _Bad()

    import labpilot.research_engine.execution.training.runner as runner_mod

    runner_mod.TrainingRunner = _CrashRunner

    result = _run(workspace)

    assert result.passed is False
    assert "KeyError: 'TVT'" in (result.error or "")
    assert result.metrics == {}


def test_empty_metrics_also_marks_the_code_suspect():
    """Reported on PR #117. There are two ways to finish with nothing — never
    writing the file, and writing one that holds no metrics — and only the
    first carried a marker, so `_training_produced_nothing` returned "" for the
    second. `code_is_suspect` stayed False and the retry reran the identical
    script to write the identical empty file."""
    from labpilot.research_engine.execution.capabilities.training.capability import (
        METRICS_EMPTY,
        METRICS_NOT_WRITTEN,
    )
    from labpilot.research_engine.execution.engineer import ResearchEngineer
    from labpilot.research_engine.planner.schemas.task_types import TaskStatus, TaskType

    class _Task:
        def __init__(self, error):
            self.type = TaskType.RUN_TRAINING
            self.status = TaskStatus.FAILED
            self.metadata = {"error": error}

    class _Plan:
        def __init__(self, error):
            self.tasks = [_Task(error)]

    for marker in (METRICS_NOT_WRITTEN, METRICS_EMPTY):
        assert ResearchEngineer._training_produced_nothing(_Plan(f"training exited 0 but {marker}"))
    assert not ResearchEngineer._training_produced_nothing(_Plan("CUDA out of memory"))


def test_a_progress_bar_cannot_crowd_out_the_traceback():
    """Reported on PR #117: `text=True` rewrites `\\r` to `\\n` before the helper
    sees it, so the documented collapse was a no-op at its own call site and a
    200-frame bar still filled the budget."""
    from labpilot.research_engine.execution.capabilities._helpers import failure_excerpt

    bar = "\n".join(f"Loading train: {i}%|#####" for i in range(300))
    traceback = "Traceback (most recent call last):\n  File \"a.py\", line 1\nKeyError: 'TVT'"

    out = failure_excerpt(bar + "\n" + traceback, "", limit=2000)

    assert out.count("Loading train") == 1
    assert "KeyError: 'TVT'" in out
    # A traceback's own consecutive lines look alike too, and every one matters.
    assert 'File "a.py"' in out


def test_two_interleaved_bars_are_not_merged():
    """Reported on PR #117: collapsing any two adjacent progress-shaped lines
    merged interleaved bars destructively — alternating `Training:` and
    `Validation:` frames became one `Validation:` line, discarding both."""
    from labpilot.research_engine.execution.capabilities._helpers import failure_excerpt

    lines = []
    for i in range(4):
        lines.append(f"Training: {i}%|##")
        lines.append(f"Validation: {i}%|##")

    out = failure_excerpt("\n".join(lines) + "\nKeyError: 'TVT'", "", limit=2000)

    assert "Training:" in out
    assert "Validation:" in out
    assert "KeyError: 'TVT'" in out


def test_one_bar_still_collapses():
    from labpilot.research_engine.execution.capabilities._helpers import failure_excerpt

    bar = "\n".join(f"Training: {i}%|##" for i in range(300))

    out = failure_excerpt(bar + "\nKeyError: 'TVT'", "", limit=2000)

    assert out.count("Training:") == 1


def test_unlabeled_bars_are_told_apart_by_their_total():
    """Reported on PR #117: tqdm's default format has no label, so two
    unrelated bars both produced an empty prefix and compared equal — the same
    state loss, gated on "no distinguishing label" instead of adjacency."""
    from labpilot.research_engine.execution.capabilities._helpers import _same_bar

    assert not _same_bar("45%|####5 | 450/1000 [00:01]", "12%|#2 | 120/2000 [00:01]")
    assert _same_bar("45%|####5 | 450/1000 [00:01]", "12%|#2 | 120/1000 [00:01]")


def test_a_fraction_in_the_label_is_not_read_as_the_bar_total():
    """Reported on PR #117: `re.search` returns the leftmost match, so an epoch
    marker in a shared label was read as the total and two loops over different
    totals collapsed into one."""
    from labpilot.research_engine.execution.capabilities._helpers import _same_bar

    assert not _same_bar(
        "Epoch 3/10 Training:  50%|## | 100/200",
        "Epoch 3/10 Training:  50%|## | 500/1000",
    )
    assert _same_bar(
        "Epoch 3/10 Training:  50%|## | 100/200",
        "Epoch 3/10 Training:  75%|###| 150/200",
    )
