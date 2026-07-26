import zipfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from labpilot.config import KaggleConfig
from labpilot.accessor.kaggle.client import KaggleClient


class FakeStatus(Enum):
    PENDING = "pending"
    COMPLETE = "complete"


class FakeSubmission(SimpleNamespace):
    pass


class FakeApi:
    def __init__(self, submission_snapshots: list[list[FakeSubmission]] | None = None) -> None:
        self.submissions: list[tuple[str, str, str]] = []
        # Each call to competition_submissions pops the next snapshot, so
        # tests can simulate Kaggle's asynchronous scoring finishing after
        # a few polls. Defaults to no history (empty list forever).
        self._snapshots = list(submission_snapshots or [])

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

    def competition_submissions(self, competition: str) -> list[FakeSubmission]:
        if not self._snapshots:
            return []
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]


def test_download_unzips_competition_files(tmp_path: Path):
    client = KaggleClient(KaggleConfig(), api=FakeApi())

    files = client.download_competition("titanic", tmp_path)

    assert files == [tmp_path / "train.csv"]
    assert not (tmp_path / "titanic.zip").exists()


def test_upload_polls_and_persists_public_score(tmp_path: Path):
    pending = FakeSubmission(description="baseline", status=FakeStatus.PENDING, public_score=None)
    complete = FakeSubmission(
        description="baseline", status=FakeStatus.COMPLETE, public_score="0.775"
    )
    api = FakeApi(submission_snapshots=[[pending], [complete]])
    config = KaggleConfig(submit_message="baseline", submission_poll_interval=0)
    client = KaggleClient(config, api=api)
    submission = tmp_path / "submission.csv"
    submission.write_text("PassengerId,Survived\n1,0\n")

    result = client.upload_submission("titanic", submission)

    assert api.submissions == [(str(submission), "baseline", "titanic")]
    assert result.status == "scored"
    assert result.public_score == 0.775


def test_fetch_competition_metadata_matches_by_ref():
    class ListingApi(FakeApi):
        def competitions_list(self, search: str) -> SimpleNamespace:
            # Kaggle's real API returns `ref` as a full URL, not a bare slug.
            competition = SimpleNamespace(
                ref="https://www.kaggle.com/competitions/titanic",
                title="Titanic - Machine Learning from Disaster",
                description="Predict survival on the Titanic.",
                category="Getting Started",
                evaluation_metric="Categorization Accuracy",
            )
            unrelated = SimpleNamespace(
                ref="https://www.kaggle.com/competitions/spaceship-titanic",
                title="Spaceship Titanic",
            )
            return SimpleNamespace(competitions=[competition, unrelated])

    client = KaggleClient(KaggleConfig(), api=ListingApi())

    metadata = client.fetch_competition_metadata("titanic")

    assert metadata is not None
    assert metadata.title == "Titanic - Machine Learning from Disaster"
    assert metadata.evaluation_metric_raw == "Categorization Accuracy"
    assert metadata.category == "Getting Started"


def test_fetch_competition_metadata_returns_none_when_no_match():
    class ListingApi(FakeApi):
        def competitions_list(self, search: str) -> SimpleNamespace:
            unrelated = SimpleNamespace(
                ref="https://www.kaggle.com/competitions/some-other-competition",
                title="Unrelated",
            )
            other = SimpleNamespace(
                ref="https://www.kaggle.com/competitions/yet-another-one",
                title="Also unrelated",
            )
            return SimpleNamespace(competitions=[unrelated, other])

    client = KaggleClient(KaggleConfig(), api=ListingApi())

    assert client.fetch_competition_metadata("titanic") is None


def test_fetch_competition_metadata_returns_none_on_error():
    class FailingApi(FakeApi):
        def competitions_list(self, search: str) -> SimpleNamespace:
            raise RuntimeError("network down")

    client = KaggleClient(KaggleConfig(), api=FailingApi())

    assert client.fetch_competition_metadata("titanic") is None


def test_upload_gives_up_after_timeout_without_score(tmp_path: Path):
    api = FakeApi(
        submission_snapshots=[
            [FakeSubmission(description="baseline", status=FakeStatus.PENDING, public_score=None)],
        ]
    )
    config = KaggleConfig(
        submit_message="baseline", submission_poll_interval=0, submission_poll_timeout=0
    )
    client = KaggleClient(config, api=api)
    submission = tmp_path / "submission.csv"
    submission.write_text("PassengerId,Survived\n1,0\n")

    result = client.upload_submission("titanic", submission)

    assert result.public_score is None
    assert result.status == "pending"
