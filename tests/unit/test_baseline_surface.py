"""§9: the verdict reaches somebody.

The gate has reached a verdict since step 5 and `build_report` had exactly one
importer — a test file. A campaign that failed produced a judgement nobody could
see, which is the same shape as the finding this whole milestone is about:
`profile.anchor_column` was correct since 2026-08-13 and nothing read it.

Two surfaces: `research baseline show` for an operator asking, and
`stop:baseline_failed` for a campaign that has to stop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from labpilot.accessor.profiler.tabular import ColumnProfile, DatasetProfile
from labpilot.cli.baseline_cli import baseline_app
from labpilot.research_engine.execution.baseline.gate import evaluate_gate
from labpilot.research_engine.execution.baseline.runner import ensure_readings

runner = CliRunner()


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
    profile = DatasetProfile(
        competition="demo",
        schema_version=4,
        target_column="y",
        row_count=n,
        train_file="train.csv",
        modalities=[{"modality": modality, "present": True, "role": "primary"}],
        columns=[
            ColumnProfile(name="x1", dtype="float64", unique_count=n, is_numeric=True),
            ColumnProfile(name="x2", dtype="float64", unique_count=n, is_numeric=True),
            ColumnProfile(
                name="y",
                dtype="float64",
                unique_count=n,
                is_numeric=True,
                stats={"min": -9.0, "max": 9.0},
            ),
        ],
    )
    (tmp_path / "profile.json").write_text(profile.model_dump_json(), encoding="utf-8")
    ensure_readings(tmp_path)
    return tmp_path


# --- research baseline show -----------------------------------------------------


def test_show_prints_every_strategy_and_the_winner(tmp_path: Path) -> None:
    """§9: "Every strategy tried and its score, the winner, the model's number,
    the verdict and what to do about it." The losers are what tell a reader
    whether a low floor means an easy target or a poor strategy."""
    workspace = _workspace(tmp_path / "ws", learnable=True)

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "mean" in result.output and "median" in result.output
    assert "lightgbm" in result.output
    assert "passed" in result.output


def test_show_prints_the_report_when_the_gate_failed(tmp_path: Path) -> None:
    """The report had one importer and it was a test file. This is the surface."""
    workspace = _workspace(tmp_path / "ws", learnable=False)
    assert evaluate_gate(workspace).state == "failed"

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.output
    assert "BASELINE FAILURE" in result.output
    assert "Do not proceed to research" in result.output


def test_show_says_what_to_do_next(tmp_path: Path) -> None:
    """Nine states exist because each has a different operator action, and a
    verdict an operator cannot act on is a wall."""
    workspace = _workspace(tmp_path / "ws", learnable=False, modality="image")

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert "awaiting_ml" in result.output
    assert "Next:" in result.output
    assert "nothing to do" in result.output
    assert "features are not columns" in result.output, "the reason, not a generic sentence"


def test_show_says_when_it_is_only_observing(tmp_path: Path) -> None:
    """A red verdict with nothing withheld has to say so, or an operator reads
    it as a stopped campaign and goes looking for what blocked it."""
    workspace = _workspace(tmp_path / "ws", learnable=False)

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert "observing only" in result.output


def test_a_path_that_is_not_a_directory_is_refused(tmp_path: Path) -> None:
    """Review finding. `--workspace /tmp/typo` reported a verdict.

    Every reader below `_root` treats absent files as absent *state*, so a path
    that does not exist read as a workspace whose campaign had not run yet: exit
    0, state `unknown`, and `Next: run research conduct` — advice that cannot
    help someone who mistyped a path.
    """
    result = runner.invoke(baseline_app, ["show", "--workspace", str(tmp_path / "nope")])

    assert result.exit_code == 2
    assert "Not a directory" in result.output
    assert "run `research conduct`" not in result.output


def test_the_refusal_says_which_situation_it_is(tmp_path: Path) -> None:
    """ "No workspace found" is the wrong sentence for a path the operator typed:
    they know where they meant, and the useful fact is that it is not there."""
    from unittest import mock

    import labpilot.cli.baseline_cli as module

    # No `create=True`. That flag is how the previous version of this test
    # passed while asserting nothing: `_root` imported `discover_workspace`
    # inside its body, so the patch invented a module attribute nothing read,
    # and the assertion held only because discovery found nothing from the
    # directory the suite happened to run in. Patching a name that must already
    # exist is what makes the mock load-bearing.
    assert hasattr(module, "discover_workspace"), "the name under test must be the one read"

    typed = runner.invoke(baseline_app, ["show", "--workspace", str(tmp_path / "nope")])
    with mock.patch.object(module, "discover_workspace", return_value=None):
        discovered = runner.invoke(baseline_app, ["show"])

    assert "Not a directory" in typed.output
    assert "No workspace found" in discovered.output


def test_discovery_is_read_through_the_name_a_test_can_patch(tmp_path: Path) -> None:
    """The mock has to reach the code, which a function-local import prevents.

    Asserted by patching discovery to *find* something and checking the command
    reports on it: if `_root` rebound the real function, this would report on
    whatever the runner's directory happens to be instead.
    """
    from unittest import mock

    import labpilot.cli.baseline_cli as module

    workspace = _workspace(tmp_path / "elsewhere", learnable=True)
    found = mock.Mock(root=workspace)

    with mock.patch.object(module, "discover_workspace", return_value=found):
        result = runner.invoke(baseline_app, ["show"])

    assert result.exit_code == 0, result.output
    assert "elsewhere" in result.output, "the patched discovery decided the workspace"


def test_waive_refuses_a_path_that_is_not_a_directory(tmp_path: Path) -> None:
    """Both commands share `_root`, so both must share the check."""
    result = runner.invoke(
        baseline_app, ["waive", "because", "--workspace", str(tmp_path / "nope")]
    )

    assert result.exit_code == 2
    assert "Not a directory" in result.output


def test_an_unmeasured_baseline_is_told_to_run_not_to_give_up(tmp_path: Path) -> None:
    """Review finding. Two situations reach `awaiting_ml` with opposite answers.

    A floor on disk and no Baseline 1 means one has not been taken yet, and
    running it is exactly the thing to do — but a lookup on the state alone said
    "a generic model cannot run on this dataset", contradicting the verdict's own
    reason two lines above it in the same output.
    """
    import json

    from labpilot.research_engine.execution.baseline.floor import FloorReading, write_floor

    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True)
    (workspace / "baseline_choice.json").write_text(
        json.dumps({"metric_name": "rmse", "target_column": "y", "validation": {"n_splits": 4}}),
        encoding="utf-8",
    )
    (workspace / "profile.json").write_text(
        DatasetProfile(competition="d", schema_version=4, target_column="y").model_dump_json(),
        encoding="utf-8",
    )
    write_floor(workspace, FloorReading(metric_name="rmse", score=1.5, best_strategy="mean"))

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert "awaiting_ml" in result.output
    assert "run the baseline" in result.output
    assert "cannot run" not in result.output


def test_a_non_terminal_state_shows_no_comparison(tmp_path: Path) -> None:
    """`awaiting_ml` never reaches the compare step, so there is nothing to show.

    Previously suppressed by `direction` happening to be an empty string, which
    `compare` then refused — an implicit dependency that a default on that field
    would have silently removed. It reads the verdict's own comparison now.
    """
    workspace = _workspace(tmp_path / "ws", learnable=False, modality="image")

    result = runner.invoke(baseline_app, ["show", "--workspace", str(workspace)])

    assert "awaiting_ml" in result.output
    assert "Improvement" not in result.output, "no comparison was ever computed"


def test_show_without_a_workspace_refuses_rather_than_guessing(tmp_path: Path) -> None:
    from unittest import mock

    import labpilot.cli.baseline_cli as module

    with mock.patch.object(module, "_root", return_value=None):
        result = runner.invoke(baseline_app, ["show"])

    assert result.exit_code == 2
    assert "No workspace found" in result.output


# --- research baseline waive -----------------------------------------------------


def test_waive_records_a_reason_and_a_fingerprint(tmp_path: Path) -> None:
    """Durable and specific, because an env-var kill switch gets set once during
    a frustrating afternoon and never unset, and nothing records that."""
    workspace = _workspace(tmp_path / "ws", learnable=False)

    result = runner.invoke(
        baseline_app, ["waive", "known-bad, shipping anyway", "--workspace", str(workspace)]
    )

    assert result.exit_code == 0, result.output
    waiver = json.loads((workspace / "baseline_waiver.json").read_text(encoding="utf-8"))
    assert waiver["reason"] == "known-bad, shipping anyway"
    assert waiver["fingerprint"], "a waiver without one applies to any dataset"
    assert evaluate_gate(workspace).state == "waived"


def test_waive_refuses_when_there_is_nothing_to_waive(tmp_path: Path) -> None:
    """Waiving a passing gate would leave a record implying a failure that never
    happened."""
    workspace = _workspace(tmp_path / "ws", learnable=True)

    result = runner.invoke(baseline_app, ["waive", "just in case", "--workspace", str(workspace)])

    assert result.exit_code == 1
    assert "Nothing to waive" in result.output
    assert not (workspace / "baseline_waiver.json").exists()


# --- stop:baseline_failed --------------------------------------------------------


def test_the_stop_is_distinct_from_failing(tmp_path: Path) -> None:
    """`failing` is a campaign whose experiments crash; this is one whose
    experiments run fine and are worse than predicting a constant. Collapsing
    them would lose the distinction this milestone exists for."""
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.conductor.loop import _baseline_failure

    workspace = _workspace(tmp_path / "ws", learnable=False)

    class _Workspace:
        root = workspace
        competition = "demo"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        stop = _baseline_failure(_Workspace())

    assert stop is not None
    rationale, report = stop
    assert rationale.startswith("stop:baseline_failed")
    assert "research baseline waive" in rationale, "a refusal must name a way out"
    assert "BASELINE FAILURE" in report


def test_observe_only_does_not_stop_the_campaign(tmp_path: Path) -> None:
    """Observe-only records the verdict and withholds nothing, which includes
    not ending the run."""
    from labpilot.research_engine.conductor.loop import _baseline_failure

    workspace = _workspace(tmp_path / "ws", learnable=False)

    class _Workspace:
        root = workspace
        competition = "demo"

    assert _baseline_failure(_Workspace()) is None


@pytest.mark.parametrize("state", ["passed", "awaiting_ml"])
def test_only_a_failed_gate_stops_the_campaign(tmp_path: Path, state: str) -> None:
    """`awaiting_ml` is a fact about the dataset. Stopping on it would end every
    image competition before it began."""
    from unittest import mock

    import labpilot.research_engine.execution.baseline.gate as gate_module
    from labpilot.research_engine.conductor.loop import _baseline_failure

    workspace = _workspace(
        tmp_path / "ws",
        learnable=state == "passed",
        modality="image" if state == "awaiting_ml" else "tabular",
    )

    class _Workspace:
        root = workspace
        competition = "demo"

    with mock.patch.object(gate_module, "enforcement_enabled", return_value=True):
        assert evaluate_gate(workspace).state == state
        assert _baseline_failure(_Workspace()) is None
