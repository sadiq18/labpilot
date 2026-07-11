import json
from pathlib import Path

from pydantic import BaseModel

from labpilot.config import KaggleConfig


class SubmissionResult(BaseModel):
    competition: str
    submission_path: str
    status: str
    public_score: float | None = None
    message: str = ""


class KaggleClient:
    """Upload submissions and fetch competition scores via Kaggle API."""

    def __init__(self, config: KaggleConfig) -> None:
        self.config = config

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult:
        # TODO: integrate kaggle.api.kaggle_api_extended.KaggleApi
        result = SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="pending",
            message=message or self.config.submit_message,
        )
        return result

    def save_result(self, run_dir: Path, result: SubmissionResult) -> Path:
        output = run_dir / "submission_result.json"
        output.write_text(result.model_dump_json(indent=2))
        return output
