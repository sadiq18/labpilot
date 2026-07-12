import pandas as pd
import pytest

from labpilot.config import AppConfig
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline

from helpers.kaggle import FakeKaggleGateway

pytestmark = pytest.mark.deep


@pytest.fixture
def deep_image_deps():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("PIL")


def test_image_deep_pipeline_generates_valid_submission(
    tmp_path,
    image_data_dir,
    competition_configs_dir,
    deep_image_deps,
):
    gateway = FakeKaggleGateway(image_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.profiler.max_images_sample = 20
    config.deep_baseline.max_epochs = 1
    config.deep_baseline.max_train_samples = 12
    config.deep_baseline.cpu_max_epochs = 1
    config.deep_baseline.cpu_max_train_samples = 12
    config.deep_baseline.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run-image-deep"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        configs_dir=competition_configs_dir,
    ).run(
        "image-deep",
        run_dir=run_dir,
    )

    assert manifest.status == StageStatus.COMPLETED
    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_accuracy" in metrics

    submission = pd.read_csv(run_dir / "submission.csv")
    assert list(submission.columns) == ["id", "label"]
