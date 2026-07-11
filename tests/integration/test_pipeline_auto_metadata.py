"""Proves the pipeline runs end-to-end for a competition with *no* local
contract file — the core "generalization" gap called out in
docs/MILESTONES.md. Metadata comes from a fake Kaggle metadata fetcher
instead, and the problem type is inferred from the profiled data, just like
it would be with a real Kaggle API response.
"""

import shutil
from pathlib import Path

from labpilot.competition.models import CompetitionMetadata
from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline


class FakeAutoResolvingGateway:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.metadata_calls: list[str] = []

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
        self.metadata_calls.append(competition)
        return CompetitionMetadata(
            slug=competition,
            title="Some Unseen Competition",
            description="A competition LabPilot has never been told about.",
            category="Playground",
            evaluation_metric_raw="Root-Mean-Squared-Error (RMSE)",
        )


def test_pipeline_generalizes_to_a_competition_without_a_local_contract(
    tmp_path: Path,
    generic_regression_data_dir: Path,
    tmp_path_factory,
):
    empty_configs_dir = tmp_path_factory.mktemp("no-contracts-here")
    gateway = FakeAutoResolvingGateway(generic_regression_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    run_dir = tmp_path / "run"

    manifest = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=empty_configs_dir,
    ).run(
        "some-unseen-competition",
        run_dir=run_dir,
    )

    assert manifest.status == StageStatus.COMPLETED
    assert gateway.metadata_calls == ["some-unseen-competition"]

    competition = (run_dir / "competition.json").read_text()
    assert "Some Unseen Competition" in competition
    assert "root_mean_squared_error" in competition or "rmse" in competition.lower()

    choice = (run_dir / "baseline_choice.json").read_text()
    assert '"problem_type":"tabular_regression"' in choice.replace(" ", "")
    # The auto-resolved evaluation metric text is informational only; the
    # actual metric key checked comes from the fixed per-problem-type
    # default, so the pipeline succeeds even though Kaggle's real metric
    # name doesn't literally match the "rmse" key the template writes.
    metrics = (run_dir / "metrics.json").read_text()
    assert "cv_rmse" in metrics
