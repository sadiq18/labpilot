import zipfile
from pathlib import Path
from types import SimpleNamespace

from labpilot.config import KaggleConfig
from labpilot.kaggle.client import KaggleClient


class FakeApi:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, str, str]] = []

    def competition_download_files(
        self,
        competition: str,
        path: str,
        force: bool,
        quiet: bool,
    ) -> None:
        destination = Path(path)
        with zipfile.ZipFile(destination / f"{competition}.zip", "w") as archive:
            archive.writestr("train.csv", "id,target\n1,0\n")

    def competition_submit(
        self,
        file_name: str,
        message: str,
        competition: str,
        quiet: bool,
    ) -> SimpleNamespace:
        self.submissions.append((file_name, message, competition))
        return SimpleNamespace(status="pending")


def test_download_unzips_competition_files(tmp_path: Path):
    client = KaggleClient(KaggleConfig(), api=FakeApi())

    files = client.download_competition("titanic", tmp_path)

    assert files == [tmp_path / "train.csv"]
    assert not (tmp_path / "titanic.zip").exists()


def test_upload_uses_official_api(tmp_path: Path):
    api = FakeApi()
    client = KaggleClient(KaggleConfig(submit_message="baseline"), api=api)
    submission = tmp_path / "submission.csv"
    submission.write_text("PassengerId,Survived\n1,0\n")

    result = client.upload_submission("titanic", submission)

    assert result.status == "pending"
    assert api.submissions == [(str(submission), "baseline", "titanic")]
