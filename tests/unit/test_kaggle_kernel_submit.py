from pathlib import Path
from types import SimpleNamespace

from labpilot.config import KaggleConfig
from labpilot.kaggle.client import KaggleClient


def test_submit_via_kernel_pushes_polls_and_scores(tmp_path: Path):
    class FakeApi:
        def __init__(self) -> None:
            self.status_polls = 0

        def kernels_push(self, folder: str) -> SimpleNamespace:
            return SimpleNamespace(
                url="https://www.kaggle.com/code/testuser/aerial-cactus-labpilot-baseline",
                versionNumber=1,
            )

        def kernels_status(self, kernel: str) -> SimpleNamespace:
            self.status_polls += 1
            return SimpleNamespace(status="COMPLETE" if self.status_polls >= 1 else "RUNNING")

        def competition_submit_code(
            self, file_name, message, competition, kernel, kernel_version, quiet=False
        ) -> None:
            pass

        def competition_submissions(self, competition: str) -> list:
            return [
                SimpleNamespace(
                    description="labpilot baseline submission",
                    public_score="0.999",
                    status=SimpleNamespace(name="COMPLETE"),
                )
            ]

    kernel_dir = tmp_path / "kernel"
    kernel_dir.mkdir()
    (kernel_dir / "submission.csv").write_text("id,pred\n1,0\n")

    config = KaggleConfig(
        submit_message="labpilot baseline submission",
        kernel_poll_interval=0,
        submission_poll_interval=0,
    )
    client = KaggleClient(config, api=FakeApi())
    result = client.submit_via_kernel("aerial-cactus-identification", kernel_dir)

    assert result.status == "scored"
    assert result.public_score == 0.999
    assert result.kernel_slug == "testuser/aerial-cactus-labpilot-baseline"
    assert result.kernel_version == 1
    assert result.submissions_url.endswith("/aerial-cactus-identification/submissions")
