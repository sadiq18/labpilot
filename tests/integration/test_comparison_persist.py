"""Integration: improve persists comparison artifacts; CLI markdown matches disk."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from labpilot.cli.main import app
from labpilot.config import AppConfig
from labpilot.experiments.comparator import load_comparison, render_markdown
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline
from helpers.kaggle import FakeKaggleGateway


def test_improve_writes_comparison_json_and_md(
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

    parent = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).run("titanic")
    assert parent.status == StageStatus.COMPLETED

    child = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).improve(parent.run_id, strategy="tune")
    assert child.status == StageStatus.COMPLETED

    child_dir = config.runs_dir / child.run_id
    assert (child_dir / "comparison.json").is_file()
    assert (child_dir / "comparison.md").is_file()

    comparison = load_comparison(child_dir)
    assert comparison is not None
    assert comparison.base_id == parent.run_id
    assert comparison.compare_id == child.run_id
    md = (child_dir / "comparison.md").read_text()
    assert "## Changes" in md
    assert "## Metrics" in md
    assert "## Conclusion" in md
    assert comparison.verdict.value in md
    assert comparison.verdict_reason in md
    assert md == render_markdown(comparison)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "compare",
            parent.run_id,
            child.run_id,
            "--format",
            "markdown",
            "--runs-dir",
            str(config.runs_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.stdout == md


def test_improve_writes_comparison_even_when_stage_fails(
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

    parent = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    ).run("titanic")

    pipeline = Pipeline(
        config, kaggle_client=gateway, configs_dir=competition_configs_dir
    )

    def boom(_run_dir, _manifest, _config):
        raise RuntimeError("forced stage failure")

    pipeline.handlers["generate_code"] = boom

    with pytest.raises(RuntimeError, match="forced stage failure"):
        pipeline.improve(parent.run_id, strategy="tune")

    children = [
        path
        for path in config.runs_dir.iterdir()
        if path.is_dir() and path.name != parent.run_id
    ]
    assert len(children) == 1
    child_dir = children[0]
    assert (child_dir / "comparison.json").is_file()
    assert (child_dir / "comparison.md").is_file()
    comparison = load_comparison(child_dir)
    assert comparison is not None
    assert comparison.base_id == parent.run_id
