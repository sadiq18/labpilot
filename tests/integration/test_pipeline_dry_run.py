import json
from pathlib import Path

from labpilot.competition.models import CompetitionSpec
from labpilot.config import AppConfig
from labpilot.kernel.exporter import build_kernel_metadata, export_kernel, slugify_kernel_id
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline

from helpers.kaggle import FakeKaggleGateway


def test_slugify_kernel_id_shortens_long_titles():
    slug = slugify_kernel_id("Aerial Cactus Identification Competition Baseline")
    assert len(slug) <= 30
    assert slug.replace("-", "").isalnum()


def test_build_kernel_metadata_uses_username_prefix():
    competition = CompetitionSpec(
        slug="aerial-cactus-identification",
        title="Aerial Cactus Identification",
        submission_mode="kernel",
    )
    kernel_id, metadata = build_kernel_metadata(competition, username="testuser")
    assert kernel_id.startswith("testuser/")
    assert metadata["id"] == kernel_id
    assert len(metadata["id"].split("/")[-1]) <= 50


def test_kernel_exporter_writes_valid_slug(tmp_path: Path):
    run_dir = tmp_path / "run"
    pipeline_dir = run_dir / "pipeline"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "train.py").write_text(
        'from pathlib import Path\n'
        'DATA_DIR = Path("/local/data/raw")\n'
        'OUTPUT_DIR = Path("/local/run")\n'
        'def main():\n'
        '    pass\n'
        'if __name__ == "__main__":\n'
        '    main()\n'
    )
    competition = CompetitionSpec(
        slug="aerial-cactus-identification",
        title="Aerial Cactus Identification",
        submission_mode="kernel",
    )

    kernel_dir = export_kernel(run_dir, competition, username="testuser")
    metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text())
    assert metadata["id"].startswith("testuser/aerial-cactus")
    assert len(metadata["id"]) < 60


def test_dry_run_produces_train_script_without_metrics(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run-dry"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
        dry_run=True,
    ).run("titanic", run_dir=run_dir)

    assert (run_dir / "pipeline" / "train.py").is_file()
    assert not (run_dir / "metrics.json").exists()
    assert (run_dir / "dry_run.json").is_file()
    train_stage = manifest.stage("train_model")
    assert train_stage is not None
    assert train_stage.status == StageStatus.SKIPPED
