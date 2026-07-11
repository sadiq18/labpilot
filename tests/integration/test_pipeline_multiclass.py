"""Proves multi-class classification support end-to-end: a synthetic 3-class,
*string*-labeled target (unlike Titanic's numeric binary target) runs through
the full pipeline and produces a valid submission.
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest

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
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="submitted",
            message=message or "test submission",
        )

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return None


def test_multiclass_pipeline_generates_valid_submission(
    tmp_path: Path, multiclass_data_dir: Path, competition_configs_dir: Path
):
    gateway = FakeKaggleGateway(multiclass_data_dir)
    config = AppConfig()
    config.training.cv_folds = 3
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run("generic-multiclass-competition", run_dir=run_dir)

    assert manifest.status == StageStatus.COMPLETED

    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_accuracy" in metrics
    assert '"num_classes": 3' in metrics

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id", "species"]
    assert len(submission) == 6
    assert set(submission["species"]).issubset({"setosa", "versicolor", "virginica"})

    oof = pd.read_csv(run_dir / "oof.csv")
    assert set(oof["prediction"]).issubset({"setosa", "versicolor", "virginica"})
    assert oof["confidence"].between(0, 1).all()


def test_numeric_multiclass_target_passes_integer_submission_validation(
    tmp_path: Path, competition_configs_dir: Path
):
    """A numeric (not string) multi-class target, e.g. digit-style class
    codes 0/1/2, should still be validated as integer labels — unlike the
    string-labeled species fixture above, where that check must be skipped.
    """
    data_dir = tmp_path / "numeric-multiclass-data"
    data_dir.mkdir()
    labels = [0, 1, 2]
    rows_per_class = 6
    train_labels = labels * rows_per_class

    train = pd.DataFrame(
        {
            "id": range(1, len(train_labels) + 1),
            "x": [i * 0.29 for i in range(len(train_labels))],
            "digit": train_labels,
        }
    )
    test = pd.DataFrame(
        {
            "id": range(len(train_labels) + 1, len(train_labels) + len(labels) + 1),
            "x": [i * 0.31 for i in range(len(labels))],
        }
    )
    sample_submission = pd.DataFrame({"id": test["id"], "digit": [0] * len(test)})

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

    (competition_configs_dir / "numeric-multiclass-competition.yaml").write_text(
        "title: Numeric Multi-Class Competition\n"
        "description: Digit-style numeric multi-class fixture.\n"
        "problem_type: tabular_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - digit\n"
    )

    gateway = FakeKaggleGateway(data_dir)
    config = AppConfig()
    config.training.cv_folds = 3
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run("numeric-multiclass-competition", run_dir=run_dir)

    assert manifest.status == StageStatus.COMPLETED
    submission = pd.read_csv(run_dir / "submission.csv")
    assert set(submission["digit"]).issubset({0, 1, 2})


@pytest.mark.parametrize("num_classes", [2, 3, 4])
def test_template_handles_varying_class_counts(
    tmp_path: Path, competition_configs_dir: Path, num_classes: int
):
    """The classification template shouldn't special-case exactly 2 classes."""
    data_dir = tmp_path / f"data-{num_classes}"
    data_dir.mkdir()
    labels = [f"class_{i}" for i in range(num_classes)]
    rows_per_class = 6
    train_labels = labels * rows_per_class

    train = pd.DataFrame(
        {
            "id": range(1, len(train_labels) + 1),
            "x": [i * 0.37 for i in range(len(train_labels))],
            "target": train_labels,
        }
    )
    test = pd.DataFrame(
        {
            "id": range(len(train_labels) + 1, len(train_labels) + len(labels) + 1),
            "x": [i * 0.41 for i in range(len(labels))],
        }
    )
    sample_submission = pd.DataFrame({"id": test["id"], "target": [labels[0]] * len(test)})

    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample_submission.to_csv(data_dir / "sample_submission.csv", index=False)

    (competition_configs_dir / "class-count-competition.yaml").write_text(
        "title: Class Count Competition\n"
        "description: Varying class-count fixture.\n"
        "problem_type: tabular_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - id\n"
        "  - target\n"
    )

    gateway = FakeKaggleGateway(data_dir)
    config = AppConfig()
    config.training.cv_folds = 3
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run("class-count-competition", run_dir=run_dir)

    assert manifest.status == StageStatus.COMPLETED
    submission = pd.read_csv(run_dir / "submission.csv")
    assert set(submission["target"]).issubset(set(labels))
