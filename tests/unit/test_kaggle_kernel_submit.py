from pathlib import Path
from types import SimpleNamespace

from labpilot.config import KaggleConfig
from labpilot.accessor.kaggle.client import KaggleClient


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


def test_kernel_ref_parses_kernels_url():
    client = KaggleClient(KaggleConfig())
    ref = client._kernel_ref_from_push(
        SimpleNamespace(),
        "https://www.kaggle.com/kernels/sadik18/aerial-cactus-labpilot-baseline",
    )
    assert ref == "sadik18/aerial-cactus-labpilot-baseline"


def test_kernel_ref_strips_code_prefix_from_push_response():
    client = KaggleClient(KaggleConfig())
    ref = client._kernel_ref_from_push(
        SimpleNamespace(ref="code/sadiq18/aerial-cactus-identification-labpilot-baseline"),
        "",
    )
    assert ref == "sadiq18/aerial-cactus-identification-labpilot-baseline"


def test_validate_push_response_rejects_error():
    client = KaggleClient(KaggleConfig())
    try:
        client._validate_push_response(SimpleNamespace(error="competition rules not accepted"))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "kernels_push failed" in str(exc)


def test_validate_push_response_rejects_empty_response():
    client = KaggleClient(KaggleConfig())
    try:
        client._validate_push_response(SimpleNamespace(error=None, url=None, versionNumber=None))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "no URL or version" in str(exc)
