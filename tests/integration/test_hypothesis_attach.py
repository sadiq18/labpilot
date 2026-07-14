from pathlib import Path

import pytest

from labpilot.config import AppConfig
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.experiments.models import HypothesisStatus
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline

from helpers.kaggle import FakeKaggleGateway


def test_pipeline_run_attaches_hypothesis_and_marks_testing(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    knowledge = tmp_path / "knowledge"
    store = HypothesisStore(knowledge, "titanic")
    hypothesis = store.create(
        observation="obs",
        reason="reason",
        prediction="pred",
        confidence=0.7,
    )
    assert hypothesis.status == HypothesisStatus.PROPOSED

    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    config.knowledge_dir = knowledge
    config.runs_dir.mkdir()

    gateway = FakeKaggleGateway(titanic_data_dir)
    manifest = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).run("titanic", hypothesis_id=hypothesis.id)

    assert manifest.status == StageStatus.COMPLETED
    assert manifest.metadata["hypothesis_id"] == hypothesis.id
    reloaded = store.get(hypothesis.id)
    assert reloaded is not None
    assert reloaded.status == HypothesisStatus.TESTING


def test_pipeline_run_rejects_missing_hypothesis(
    tmp_path: Path,
    titanic_data_dir: Path,
    competition_configs_dir: Path,
):
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"
    config.knowledge_dir = tmp_path / "knowledge"
    config.runs_dir.mkdir()

    gateway = FakeKaggleGateway(titanic_data_dir)
    with pytest.raises(FileNotFoundError, match="H-404"):
        Pipeline(
            config, kaggle_client=gateway, configs_dir=competition_configs_dir
        ).run("titanic", hypothesis_id="H-404")
