import shutil
from pathlib import Path

import pandas as pd

from labpilot.competition.models import CompetitionMetadata
from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline


class FakeKaggleGateway:
    def __init__(self, source: Path) -> None:
        self.source = source

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
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


def test_regression_pipeline_generates_valid_submission(
    tmp_path: Path,
    generic_regression_data_dir: Path,
    competition_configs_dir: Path,
):
    """Guards the regression baseline path: target/ID inference from the
    contract (not a "last column" guess), fold-safe categorical encoding,
    version-safe RMSE, and non-integer submission validation.
    """
    gateway = FakeKaggleGateway(generic_regression_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=competition_configs_dir,
    ).run(
        "generic-regression-competition",
        run_dir=run_dir,
    )

    assert manifest.status == StageStatus.COMPLETED

    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_rmse" in metrics

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id", "target"]
    assert len(submission) == 5
    # Regression targets are continuous; this would fail if the pipeline
    # incorrectly applied the classification-only integer-label check.
    assert not (submission["target"] % 1 == 0).all()
