import shutil
from pathlib import Path

import pandas as pd
import pytest

from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline


class FakeKaggleGateway:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.uploads: list[Path] = []

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for source_file in self.source.glob("*.csv"):
            destination_file = destination / source_file.name
            shutil.copy2(source_file, destination_file)
            files.append(destination_file)
        return sorted(files)

    def upload_submission(
        self,
        competition: str,
        submission_path: Path,
        message: str | None = None,
    ) -> SubmissionResult:
        self.uploads.append(submission_path)
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="submitted",
            message=message or "test submission",
        )


@pytest.mark.parametrize("submit", [False, True])
def test_titanic_pipeline_generates_valid_submission(
    tmp_path: Path,
    titanic_data_dir: Path,
    submit: bool,
):
    gateway = FakeKaggleGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    run_dir = tmp_path / f"run-{submit}"

    manifest = Pipeline(config, kaggle_client=gateway, submit=submit).run(
        "titanic",
        run_dir=run_dir,
    )

    assert manifest.status == StageStatus.COMPLETED
    expected_upload_status = StageStatus.COMPLETED if submit else StageStatus.SKIPPED
    assert manifest.stage("upload_submission").status == expected_upload_status
    assert len(gateway.uploads) == int(submit)

    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_accuracy" in metrics

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["PassengerId", "Survived"]
    assert len(submission) == 4
    assert set(submission["Survived"]).issubset({0, 1})
