"""Proves the two-phase `research init` / `research build` workflow: init
stops after the brief (competition/data/profile/brief only), and build picks
up from there through the rest of the pipeline — without redoing anything
init already finished.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from labpilot.competition.models import CompetitionMetadata
from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import INIT_STAGES, Pipeline


class FakeKaggleGateway:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.download_calls = 0

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        self.download_calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for source_file in self.source.glob("*.csv"):
            destination_file = destination / source_file.name
            shutil.copy2(source_file, destination_file)
            files.append(destination_file)
        return sorted(files)

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult:
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="submitted",
            message=message or "test submission",
        )

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return None


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    return config


def test_init_stops_after_the_brief(
    tmp_path: Path, titanic_data_dir: Path, competition_configs_dir: Path
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = _config(tmp_path)
    pipeline = Pipeline(config, kaggle_client=gateway, configs_dir=competition_configs_dir)

    manifest = pipeline.init("titanic")

    assert manifest.status == StageStatus.PARTIAL
    assert {record.name for record in manifest.stages} == set(INIT_STAGES)
    assert all(record.status == StageStatus.COMPLETED for record in manifest.stages)

    run_dir = config.runs_dir / manifest.run_id
    assert (run_dir / "brief.md").exists()
    assert not (run_dir / "baseline_choice.json").exists()
    assert not (run_dir / "metrics.json").exists()


def test_build_continues_and_finishes_the_pipeline(
    tmp_path: Path, titanic_data_dir: Path, competition_configs_dir: Path
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = _config(tmp_path)
    pipeline = Pipeline(config, kaggle_client=gateway, configs_dir=competition_configs_dir)

    init_manifest = pipeline.init("titanic")
    built_manifest = pipeline.build(init_manifest.run_id)

    assert built_manifest.status == StageStatus.COMPLETED
    all_names = {record.name for record in built_manifest.stages}
    assert all_names == set(pipeline.handlers.keys())

    run_dir = config.runs_dir / init_manifest.run_id
    assert (run_dir / "submission.csv").exists()
    assert (run_dir / "reflection.md").exists()
    assert (run_dir / "report.html").exists()

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["PassengerId", "Survived"]


def test_build_refuses_to_run_before_init_finished(tmp_path: Path, competition_configs_dir: Path):
    config = _config(tmp_path)
    pipeline = Pipeline(config, configs_dir=competition_configs_dir)

    # A manifest that only got through parse_competition, not the rest of init.
    started = pipeline._start("titanic", None)
    manifest, run_dir, _all_stages = started
    from labpilot.orchestrator.manifest import save_manifest

    manifest.mark_completed("parse_competition", [])
    save_manifest(run_dir, manifest)

    with pytest.raises(ValueError, match="hasn't finished its init stage"):
        pipeline.build(manifest.run_id)
