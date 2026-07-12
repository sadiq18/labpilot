from datetime import datetime, timedelta

import pytest

from labpilot.competition.models import CompetitionSpec
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.pipeline import Pipeline
from labpilot.config import AppConfig, KaggleConfig


class FakeKaggleWithQuota:
    def download_competition(self, competition, destination):
        return []

    def upload_submission(self, competition, submission_path, message=None):
        return SubmissionResult(competition=competition, submission_path=str(submission_path), status="submitted")

    def fetch_competition_metadata(self, competition):
        return None

    def count_todays_submissions(self, competition: str) -> int:
        return 5


def test_preflight_blocks_disabled_submissions():
    pipeline = Pipeline(AppConfig(), kaggle_client=FakeKaggleWithQuota(), submit=True)
    competition = CompetitionSpec(slug="test", submissions_disabled=True)
    with pytest.raises(ValueError, match="disabled"):
        pipeline._preflight_submission(competition)


def test_preflight_blocks_kernels_only():
    pipeline = Pipeline(AppConfig(), kaggle_client=FakeKaggleWithQuota(), submit=True)
    competition = CompetitionSpec(slug="test", is_kernels_submissions_only=True)
    with pytest.raises(ValueError, match="kernels-only"):
        pipeline._preflight_submission(competition)


def test_preflight_blocks_past_deadline():
    pipeline = Pipeline(AppConfig(), kaggle_client=FakeKaggleWithQuota(), submit=True)
    past = (datetime.now() - timedelta(days=1)).isoformat()
    competition = CompetitionSpec(slug="test", deadline=past)
    with pytest.raises(ValueError, match="deadline"):
        pipeline._preflight_submission(competition)


def test_preflight_allows_past_deadline_with_force_submit():
    pipeline = Pipeline(
        AppConfig(), kaggle_client=FakeKaggleWithQuota(), submit=True, force_submit=True
    )
    past = (datetime.now() - timedelta(days=1)).isoformat()
    competition = CompetitionSpec(slug="test", deadline=past)
    pipeline._preflight_submission(competition)


def test_preflight_blocks_daily_quota():
    pipeline = Pipeline(AppConfig(), kaggle_client=FakeKaggleWithQuota(), submit=True)
    competition = CompetitionSpec(slug="test", max_daily_submissions=5)
    with pytest.raises(ValueError, match="quota"):
        pipeline._preflight_submission(competition)
