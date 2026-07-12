import pandas as pd

from labpilot.config import AppConfig
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline

from helpers.kaggle import FakeKaggleGateway


def test_text_pipeline_generates_valid_submission(
    tmp_path,
    text_data_dir,
    competition_configs_dir,
):
    gateway = FakeKaggleGateway(text_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run-text"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run(
        "text-sentiment",
        run_dir=run_dir,
    )

    assert manifest.status == StageStatus.COMPLETED
    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_accuracy" in metrics

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id", "label"]
    assert len(submission) == 4
