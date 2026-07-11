import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from labpilot.config import KaggleConfig

logger = logging.getLogger(__name__)


class SubmissionResult(BaseModel):
    competition: str
    submission_path: str
    status: str
    public_score: float | None = None
    message: str = ""


class KaggleGateway(Protocol):
    """Network boundary used by the pipeline and its tests."""

    def download_competition(self, competition: str, destination: Path) -> list[Path]: ...

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult: ...


class KaggleClient:
    """Download data and upload submissions through Kaggle's official API.

    # TODO: control the verbosity of this class's logging via a future CLI
    # --verbose/--quiet flag (see docs/MILESTONES.md).
    """

    def __init__(self, config: KaggleConfig, api: Any | None = None) -> None:
        self.config = config
        self._api = api

    def _configure_environment(self) -> None:
        if self.config.api_token:
            os.environ["KAGGLE_API_TOKEN"] = self.config.api_token
        if self.config.username:
            os.environ["KAGGLE_USERNAME"] = self.config.username
        if self.config.key:
            os.environ["KAGGLE_KEY"] = self.config.key

    def authenticate(self) -> Any:
        if self._api is not None:
            return self._api

        logger.info("Authenticating with the Kaggle API.")
        self._configure_environment()
        auth_error = RuntimeError(
            "Kaggle authentication failed. Set KAGGLE_API_TOKEN, or configure "
            "~/.kaggle/access_token (legacy KAGGLE_USERNAME/KAGGLE_KEY also work)."
        )
        try:
            # kaggle>=2.0 checks for credentials as soon as this module is
            # imported and calls sys.exit(1) (raising SystemExit, not a
            # regular Exception) if none are configured, so both the import
            # and the explicit authenticate() call below need to guard
            # against SystemExit, not just Exception/ImportError.
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError as exc:
            raise RuntimeError(
                'Kaggle support is not installed. Run: pip install -e ".[dev,llm]"'
            ) from exc
        except SystemExit as exc:
            raise auth_error from exc

        api = KaggleApi()
        try:
            api.authenticate()
        except (Exception, SystemExit) as exc:
            raise auth_error from exc
        self._api = api
        logger.info("Kaggle authentication succeeded.")
        return api

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        logger.info("Downloading competition '%s' into %s", competition, destination)
        destination.mkdir(parents=True, exist_ok=True)
        api = self.authenticate()
        try:
            api.competition_download_files(
                competition,
                path=str(destination),
                force=True,
                quiet=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download competition '{competition}'. Confirm the slug and "
                "accept the competition rules on Kaggle."
            ) from exc

        if self.config.download_unzip:
            for archive in destination.glob("*.zip"):
                logger.info("Extracting %s", archive)
                with zipfile.ZipFile(archive) as zipped:
                    zipped.extractall(destination)
                archive.unlink()

        files = sorted(path for path in destination.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Kaggle returned no files for competition '{competition}'.")
        logger.info("Downloaded %d file(s) for '%s'.", len(files), competition)
        return files

    def upload_submission(
        self, competition: str, submission_path: Path, message: str | None = None
    ) -> SubmissionResult:
        if not submission_path.is_file():
            raise FileNotFoundError(f"Submission file not found: {submission_path}")

        submission_message = message or self.config.submit_message
        logger.info("Uploading %s to '%s'.", submission_path, competition)
        api = self.authenticate()
        try:
            response = api.competition_submit(
                str(submission_path),
                submission_message,
                competition,
                quiet=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to submit to '{competition}'. Confirm that competition rules "
                "are accepted and the submission quota is available."
            ) from exc

        status = getattr(response, "status", None) or "submitted"
        logger.info("Submission to '%s' completed with status '%s'.", competition, status)

        public_score = self._poll_public_score(api, competition, submission_message)
        result_status = "scored" if public_score is not None else str(status)
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status=result_status,
            public_score=public_score,
            message=submission_message,
        )

    def _poll_public_score(self, api: Any, competition: str, message: str) -> float | None:
        """Poll for the public leaderboard score of the submission just made.

        Kaggle scores submissions asynchronously, so the response from
        `competition_submit` never carries a score. We poll
        `competition_submissions` (newest first) for the matching submission
        until it finishes scoring or `submission_poll_timeout` elapses.
        """
        deadline = time.monotonic() + self.config.submission_poll_timeout
        while True:
            try:
                submissions = api.competition_submissions(competition)
            except Exception:
                logger.warning("Unable to poll submission status for '%s'.", competition)
                return None

            latest = submissions[0] if submissions else None
            if latest is not None and getattr(latest, "description", None) == message:
                score = getattr(latest, "public_score", None)
                if score not in (None, ""):
                    try:
                        return float(score)
                    except (TypeError, ValueError):
                        return None
                status_name = getattr(getattr(latest, "status", None), "name", None)
                if status_name and status_name not in ("PENDING", "SUBMISSION_STATUS_UNSPECIFIED"):
                    # Finished (e.g. COMPLETE/ERROR) without a score to report.
                    return None

            if time.monotonic() >= deadline:
                logger.info(
                    "Timed out after %ss waiting for '%s' to finish scoring.",
                    self.config.submission_poll_timeout,
                    competition,
                )
                return None
            time.sleep(self.config.submission_poll_interval)

    @staticmethod
    def save_result(run_dir: Path, result: SubmissionResult) -> Path:
        output = run_dir / "submission_result.json"
        output.write_text(result.model_dump_json(indent=2))
        return output
