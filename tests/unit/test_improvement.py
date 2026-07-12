from pathlib import Path

import pytest

from labpilot.improvement.fork import COPY_FILES, fork_run
from labpilot.improvement.models import (
    ImprovementAction,
    ImprovementPlan,
    TrainingOverrides,
    load_training_overrides,
    save_improvement_plan,
    save_training_overrides,
)
from labpilot.config import LLMConfig
from labpilot.improvement.planner import ImprovementPlanner
from labpilot.improvement.tuner import grid_combinations, pick_tune_params
from labpilot.orchestrator.manifest import RunManifest, StageStatus, save_manifest


def _completed_parent(tmp_path: Path, run_id: str = "20260101-120000-titanic") -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "competition.json").write_text('{"slug": "titanic"}')
    (run_dir / "profile.json").write_text('{"competition": "titanic", "columns": []}')
    (run_dir / "profile.md").write_text("# profile")
    (run_dir / "brief.md").write_text("# brief")
    (run_dir / "baseline_choice.json").write_text(
        '{"problem_type": "tabular_classification", "template_name": "tabular_classification", '
        '"rationale": "test", "target_column": "Survived", "id_column": "PassengerId", '
        '"train_file": "train.csv", "test_file": "test.csv", '
        '"sample_submission_file": "gender_submission.csv", "metric_name": "accuracy"}'
    )
    data_dir = run_dir / "data" / "raw"
    data_dir.mkdir(parents=True)
    (data_dir / "train.csv").write_text("a\n1")

    manifest = RunManifest(
        run_id=run_id,
        competition="titanic",
        status=StageStatus.COMPLETED,
        stages=[],
    )
    for stage in (
        "parse_competition",
        "download_data",
        "profile_dataset",
        "generate_brief",
        "select_baseline",
        "generate_code",
        "train_model",
        "evaluate_cv",
        "generate_submission",
        "log_experiment",
        "write_reflection",
    ):
        manifest.mark_completed(stage, [])
    save_manifest(run_dir, manifest)
    return run_dir


def test_fork_copies_init_artifacts_and_lineage(tmp_path: Path):
    parent_dir = _completed_parent(tmp_path)
    child_id, child_dir = fork_run(parent_dir, tmp_path, improvement_strategy="tune")

    assert child_id != parent_dir.name
    assert child_dir.is_dir()
    for name in COPY_FILES:
        assert (child_dir / name).is_file()
    assert (child_dir / "data" / "raw" / "train.csv").is_file()

    manifest = RunManifest.model_validate_json((child_dir / "manifest.json").read_text())
    assert manifest.metadata["parent_run_id"] == parent_dir.name
    assert manifest.metadata["iteration"] == 1
    assert manifest.metadata["improvement_strategy"] == "tune"
    assert manifest.stage("generate_brief").status == StageStatus.COMPLETED
    assert manifest.stage("select_baseline").status == StageStatus.COMPLETED
    generate_code = manifest.stage("generate_code")
    assert generate_code is None or generate_code.status == StageStatus.PENDING


def test_fork_requires_completed_parent(tmp_path: Path):
    run_dir = tmp_path / "partial-run"
    run_dir.mkdir()
    manifest = RunManifest(
        run_id="partial-run",
        competition="titanic",
        status=StageStatus.PARTIAL,
        stages=[],
    )
    save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="must be completed"):
        fork_run(run_dir, tmp_path)


def test_tune_planner_picks_different_params(tmp_path: Path):
    parent_dir = _completed_parent(tmp_path)
    (parent_dir / "training_overrides.json").write_text(
        TrainingOverrides(
            model_params={"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 300}
        ).model_dump_json()
    )

    planner = ImprovementPlanner(config=LLMConfig())
    plan, overrides = planner.plan(parent_dir, parent_dir.name, strategy="tune", random_seed=42)

    assert plan.strategy == "tune"
    assert ImprovementAction.TUNE_HYPERPARAMS in plan.actions
    assert overrides.model_params != {}
    assert overrides.model_params.get("random_state") == 42


def test_pick_tune_params_advances_grid():
    parent = {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 300}
    next_params = pick_tune_params(parent, random_seed=7)
    assert next_params != parent
    assert len(grid_combinations()) <= 12


def test_training_overrides_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    overrides = TrainingOverrides(
        model_params={"learning_rate": 0.1},
        feature_recipes=["log_numeric"],
        log_numeric_columns=["Fare"],
    )
    save_training_overrides(run_dir, overrides)
    loaded = load_training_overrides(run_dir)
    assert loaded.model_params["learning_rate"] == 0.1
    assert loaded.feature_recipes == ["log_numeric"]


def test_improvement_plan_persisted(tmp_path: Path):
    run_dir = tmp_path / "child"
    run_dir.mkdir()
    plan = ImprovementPlan(
        parent_run_id="parent",
        strategy="tune",
        actions=[ImprovementAction.TUNE_HYPERPARAMS],
        model_params={"learning_rate": 0.03},
        rationale="test",
    )
    path = save_improvement_plan(run_dir, plan)
    assert path.is_file()
    assert "improvement_plan.json" in path.name
