import json
from pathlib import Path

from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.config import AppConfig
from labpilot.experiments.graph import assemble_experiment, build_graph
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline
from labpilot.tracking.index import diff_runs, scan_runs

from helpers.kaggle import FakeKaggleGateway


def test_experiment_graph_smoke_end_to_end(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    """End-to-end smoke test for Milestone 2, Plan 1 (Experiment Graph).

    Exercises every piece of the changeset in one pass, against a real
    (fake-gateway) pipeline run + improve — not a hand-seeded fixture:

    1. The two writers: `Pipeline._start()` and `fork_run()` (`config.json`
       + `git_commit`).
    2. `experiments/graph.py`: `assemble_experiment`, `build_graph`,
       `ExperimentGraph` (roots/children/ancestors/best_path).
    3. `tracking/index.py`: `scan_runs()` (refactored onto `graph.py`) and
       `diff_runs()` still work unchanged.
    4. The CLI: `research experiments graph` and `research experiments
       show` (table + json).
    """
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    config.runs_dir.mkdir(parents=True)

    parent_manifest = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).run("titanic")
    assert parent_manifest.status == StageStatus.COMPLETED

    child_manifest = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).improve(parent_manifest.run_id, strategy="tune")
    assert child_manifest.status == StageStatus.COMPLETED

    parent_id = parent_manifest.run_id
    child_id = child_manifest.run_id
    parent_dir = config.runs_dir / parent_id
    child_dir = config.runs_dir / child_id

    # 1. Writers: config.json + git_commit for both parent and child.
    assert (parent_dir / "config.json").is_file()
    assert (child_dir / "config.json").is_file()
    assert "git_commit" in parent_manifest.metadata
    assert "git_commit" in child_manifest.metadata

    # 2. experiments/graph.py: single-run assembly.
    parent_experiment = assemble_experiment(parent_dir)
    child_experiment = assemble_experiment(child_dir)
    assert parent_experiment.parent_id is None
    assert child_experiment.parent_id == parent_id
    assert child_experiment.iteration == 1
    assert parent_experiment.description  # non-empty, no LLM call
    assert child_experiment.description
    assert parent_experiment.progress.endswith(" stages")
    assert parent_experiment.config_snapshot  # non-empty round-tripped dict
    assert any(path.endswith("submission.csv") for path in parent_experiment.artifacts)
    assert any(path.endswith("config.json") for path in parent_experiment.artifacts)

    # 2b. experiments/graph.py: full parent/child graph.
    graph = build_graph(config.runs_dir, "titanic")
    assert {exp.id for exp in graph.roots} == {parent_id}
    assert {exp.id for exp in graph.children(parent_id)} == {child_id}
    assert [exp.id for exp in graph.ancestors(child_id)] == [parent_id]
    assert {exp.id for exp in graph.descendants(parent_id)} == {child_id}

    best_path = graph.best_path("cv_accuracy")
    assert [exp.id for exp in best_path][0] == parent_id

    # 3. tracking/index.py: scan_runs()/diff_runs() unaffected by the refactor.
    entries = scan_runs(config.runs_dir)
    assert {entry.run_id for entry in entries} == {parent_id, child_id}

    diff = diff_runs(config.runs_dir, parent_id, child_id)
    assert diff.compare_params.get("parent_run_id") == parent_id

    # 4. CLI: `research experiments graph` / `research experiments show`.
    runner = CliRunner()

    graph_result = runner.invoke(
        app,
        [
            "experiments",
            "graph",
            "--competition",
            "titanic",
            "--metric",
            "cv_accuracy",
            "--runs-dir",
            str(config.runs_dir),
        ],
    )
    assert graph_result.exit_code == 0, graph_result.output
    assert parent_id in graph_result.output
    assert child_id in graph_result.output

    show_result = runner.invoke(
        app,
        ["experiments", "show", parent_id, "--runs-dir", str(config.runs_dir)],
    )
    assert show_result.exit_code == 0, show_result.output
    assert "completed" in show_result.output

    show_json_result = runner.invoke(
        app,
        [
            "experiments",
            "show",
            child_id,
            "--format",
            "json",
            "--runs-dir",
            str(config.runs_dir),
        ],
    )
    assert show_json_result.exit_code == 0, show_json_result.output
    payload = json.loads(show_json_result.output)
    assert payload["id"] == child_id
    assert payload["parent_id"] == parent_id
    assert payload["git_commit"] is None or isinstance(payload["git_commit"], str)
