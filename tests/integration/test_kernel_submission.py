"""Integration tests for kernel-only submission workflow."""

import shutil
from pathlib import Path
from types import SimpleNamespace

from labpilot.competition.models import CompetitionMetadata
from labpilot.config import AppConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.kaggle.urls import competition_submissions_url, kernel_notebook_url
from labpilot.orchestrator.manifest import StageStatus, load_manifest
from labpilot.orchestrator.pipeline import Pipeline


class KernelGateway:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.kernel_submits: list[tuple[str, Path]] = []

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for source_file in self.source.glob("*.csv"):
            dest_file = destination / source_file.name
            shutil.copy2(source_file, dest_file)
            files.append(dest_file)
        return sorted(files)

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult:
        raise NotImplementedError("CSV upload should not run for kernel competitions")

    def submit_via_kernel(
        self,
        competition: str,
        kernel_dir: Path,
        *,
        output_file: str = "submission.csv",
        message: str | None = None,
        existing_kernel_slug: str | None = None,
        existing_kernel_version: int | None = None,
    ) -> SubmissionResult:
        self.kernel_submits.append((competition, kernel_dir))
        return SubmissionResult(
            competition=competition,
            submission_path=str(kernel_dir / output_file),
            status="scored",
            public_score=0.91,
            message=message or "kernel submission",
            submission_mode="kernel",
            kernel_slug="testuser/aerial-cactus-labpilot-baseline",
            kernel_version=3,
            kernel_run_status="COMPLETE",
            submissions_url=competition_submissions_url(competition),
            kernel_url=kernel_notebook_url("testuser", "aerial-cactus-labpilot-baseline", 3),
        )

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return CompetitionMetadata(
            slug=competition,
            title="Kernel Competition",
            is_kernels_submissions_only=True,
        )

    def count_todays_submissions(self, competition: str) -> int:
        return 0


def test_kernel_pipeline_exports_and_submits(
    tmp_path: Path,
    titanic_data_dir: Path,
):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "kernel-comp.yaml").write_text(
        "title: Kernel Competition\n"
        "problem_type: tabular_classification\n"
        "evaluation_metric:\n"
        "  name: accuracy\n"
        "  direction: maximize\n"
        "submission_columns:\n"
        "  - PassengerId\n"
        "  - Survived\n"
        "is_kernels_submissions_only: true\n"
        "submission_mode: kernel\n"
    )

    gateway = KernelGateway(titanic_data_dir)
    config = AppConfig()
    config.training.cv_folds = 2
    config.kaggle.cache_dir = tmp_path / "kaggle-cache"
    config.runs_dir = tmp_path / "runs"

    pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=False,
        configs_dir=configs_dir,
    )
    manifest = pipeline.run("kernel-comp")

    assert manifest.status == StageStatus.COMPLETED
    assert manifest.stage("export_kernel").status == StageStatus.COMPLETED
    assert manifest.stage("upload_submission").status == StageStatus.SKIPPED
    assert gateway.kernel_submits == []

    run_dir = config.runs_dir / manifest.run_id
    assert (run_dir / "kernel" / "run.py").is_file()
    result = SubmissionResult.model_validate_json((run_dir / "submission_result.json").read_text())
    assert result.status == "kernel_ready"
    assert result.submission_mode == "kernel"

    submit_pipeline = Pipeline(
        config,
        kaggle_client=gateway,
        submit=True,
        configs_dir=configs_dir,
    )
    resumed = submit_pipeline.resume(manifest.run_id)

    assert resumed.stage("upload_submission").status == StageStatus.COMPLETED
    assert len(gateway.kernel_submits) == 1
    result = SubmissionResult.model_validate_json((run_dir / "submission_result.json").read_text())
    assert result.status == "scored"
    assert result.kernel_slug == "testuser/aerial-cactus-labpilot-baseline"
    assert "submissions" in (result.submissions_url or "")
    reflection = (run_dir / "reflection.md").read_text()
    assert "## Submission links" in reflection


def test_submit_via_kernel_retries_code_submit_without_repush(tmp_path: Path):
    from labpilot.config import KaggleConfig
    from labpilot.kaggle.client import KaggleClient

    class FakeApi:
        def __init__(self) -> None:
            self.push_calls = 0
            self.code_submits: list[tuple] = []

        def kernels_push(self, folder: str) -> SimpleNamespace:
            self.push_calls += 1
            raise AssertionError("should not re-push on retry")

        def kernels_status(self, kernel: str) -> SimpleNamespace:
            return SimpleNamespace(status="COMPLETE")

        def competition_submit_code(
            self, file_name, message, competition, kernel, kernel_version, quiet=False
        ) -> None:
            self.code_submits.append((file_name, competition, kernel, kernel_version))

        def competition_submissions(self, competition: str) -> list:
            return [
                SimpleNamespace(
                    description="labpilot baseline submission",
                    public_score="0.88",
                    status=SimpleNamespace(name="COMPLETE"),
                )
            ]

    api = FakeApi()
    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "submission.csv").write_text("id,pred\n1,0\n")

    client = KaggleClient(
        KaggleConfig(submit_message="labpilot baseline submission", submission_poll_interval=0),
        api=api,
    )
    result = client.submit_via_kernel(
        "kernel-comp",
        kernel_dir,
        existing_kernel_slug="user/kernel-slug",
        existing_kernel_version=2,
    )

    assert api.push_calls == 0
    assert len(api.code_submits) == 1
    assert result.status == "scored"
    assert result.public_score == 0.88
    assert result.kernel_slug == "user/kernel-slug"
