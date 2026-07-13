from pathlib import Path

from labpilot.config import AppConfig
from labpilot.improvement.models import load_training_overrides
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline
from labpilot.tracking.index import diff_runs
from helpers.kaggle import FakeKaggleGateway


def test_improve_tune_creates_child_with_different_params(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    config.runs_dir.mkdir(parents=True)

    parent_manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run("titanic")

    assert parent_manifest.status == StageStatus.COMPLETED

    # Mutate an inert config field (unused by training/kaggle stages) before
    # improving, so the parent/child config.json snapshots below can only
    # match if each run captured its own config at its own point in time
    # rather than one being copied from the other.
    config.llm.model = "gpt-4o-mini-child"

    child_manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).improve(parent_manifest.run_id, strategy="tune")

    assert child_manifest.status == StageStatus.COMPLETED
    assert child_manifest.metadata["parent_run_id"] == parent_manifest.run_id
    assert child_manifest.metadata["iteration"] == 1

    child_dir = config.runs_dir / child_manifest.run_id
    assert (child_dir / "improvement_plan.json").is_file()
    assert (child_dir / "training_overrides.json").is_file()
    assert (child_dir / "metrics.json").is_file()

    overrides = load_training_overrides(child_dir)
    assert overrides.model_params

    diff = diff_runs(config.runs_dir, parent_manifest.run_id, child_manifest.run_id)
    assert diff.base_run_id == parent_manifest.run_id
    assert diff.compare_run_id == child_manifest.run_id
    assert diff.compare_params.get("parent_run_id") == parent_manifest.run_id

    # Milestone 2, Plan 1: Research Memory — every run gets its own config
    # snapshot, and the child's is independently resolved, not copied from
    # the parent's.
    parent_dir = config.runs_dir / parent_manifest.run_id
    assert (parent_dir / "config.json").is_file()
    assert (child_dir / "config.json").is_file()

    parent_config = AppConfig.model_validate_json((parent_dir / "config.json").read_text())
    child_config = AppConfig.model_validate_json((child_dir / "config.json").read_text())
    assert parent_config.llm.model == "gpt-4o-mini"
    assert child_config.llm.model == "gpt-4o-mini-child"
