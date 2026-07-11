"""Proves `--resume` continues a run from its first failed stage instead of
starting over — the "restart from failed stage" item in docs/MILESTONES.md.
"""

import shutil
from pathlib import Path

import pytest

from labpilot.competition.models import CompetitionMetadata
from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import StageStatus, load_manifest
from labpilot.orchestrator.pipeline import Pipeline


class FlakyGateway:
    """Fails `download_competition` a fixed number of times, then succeeds."""

    def __init__(self, source: Path, fail_times: int = 1) -> None:
        self.source = source
        self.fail_times = fail_times
        self.download_calls = 0

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        self.download_calls += 1
        if self.download_calls <= self.fail_times:
            raise RuntimeError("simulated transient network failure")

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
        raise NotImplementedError

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return None


def test_resume_continues_from_the_failed_stage(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    gateway = FlakyGateway(titanic_data_dir, fail_times=1)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"

    pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=competition_configs_dir,
    )

    with pytest.raises(RuntimeError):
        pipeline.run("titanic")

    run_id = next(p.name for p in config.runs_dir.iterdir())
    manifest = load_manifest(config.runs_dir / run_id)
    assert manifest.status == StageStatus.FAILED
    assert manifest.stage("parse_competition").status == StageStatus.COMPLETED
    assert manifest.stage("download_data").status == StageStatus.FAILED
    assert manifest.stage("profile_dataset") is None

    resumed = pipeline.resume(run_id)

    assert resumed.status == StageStatus.COMPLETED
    assert gateway.download_calls == 2
    # parse_competition was already done and must not have been re-run.
    assert resumed.stage("parse_competition").status == StageStatus.COMPLETED


def test_resume_on_a_completed_run_is_a_no_op(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    gateway = FlakyGateway(titanic_data_dir, fail_times=0)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"

    pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=competition_configs_dir,
    )
    manifest = pipeline.run("titanic")

    resumed = pipeline.resume(manifest.run_id)

    assert resumed.status == StageStatus.COMPLETED
    assert gateway.download_calls == 1


class UploadingGateway(FlakyGateway):
    """A `FlakyGateway` whose `upload_submission` actually records uploads,
    instead of raising `NotImplementedError` like the base fixture (which
    assumes `--submit` is never used).
    """

    def __init__(self, source: Path) -> None:
        super().__init__(source, fail_times=0)
        self.uploads: list[Path] = []

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult:
        self.uploads.append(submission_path)
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="submitted",
            message=message or "test submission",
        )


def test_resume_with_submit_uploads_a_previously_skipped_submission(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    """A run finished without `--submit` marks `upload_submission` as
    `skipped`, not `completed` — passing `--submit` on a later `resume` (or
    `build`) call for that same run must actually upload this time, rather
    than treating the earlier `skipped` status as "nothing left to do".
    """
    gateway = UploadingGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"

    first_pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=competition_configs_dir,
    )
    manifest = first_pipeline.run("titanic")

    assert manifest.status == StageStatus.COMPLETED
    assert manifest.stage("upload_submission").status == StageStatus.SKIPPED
    assert gateway.uploads == []

    resubmit_pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=True,
        configs_dir=competition_configs_dir,
    )
    resumed = resubmit_pipeline.resume(manifest.run_id)

    assert resumed.status == StageStatus.COMPLETED
    assert resumed.stage("upload_submission").status == StageStatus.COMPLETED
    assert len(gateway.uploads) == 1
    # Every other stage was already done and must not have been re-run.
    assert gateway.download_calls == 1
