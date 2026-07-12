import logging
import os
import time
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from labpilot.competition.models import CompetitionMetadata
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

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None: ...

    def count_todays_submissions(self, competition: str) -> int: ...


class KaggleClient:
    """Download data and upload submissions through Kaggle's official API."""

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

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        """Best-effort lookup of a competition's title/description/metric.

        Used by `CompetitionParser` to auto-resolve a competition contract
        when no local override file exists. Returns None (rather than
        raising) on any failure — auth issues, no network, or an unknown
        slug — since this is an enhancement, not a hard requirement; the
        parser falls back to a bare-minimum contract when this is None.
        """
        try:
            api = self.authenticate()
            response = api.competitions_list(search=competition)
        except Exception:
            logger.warning(
                "Could not resolve metadata for '%s' from the Kaggle API.",
                competition,
                exc_info=True,
            )
            return None

        candidates = list(getattr(response, "competitions", None) or [])
        if not candidates:
            return None

        match = next(
            (c for c in candidates if self._ref_slug(getattr(c, "ref", "")) == competition.lower()),
            candidates[0] if len(candidates) == 1 else None,
        )
        if match is None:
            logger.info(
                "Kaggle search for '%s' returned %d ambiguous result(s); skipping auto-metadata.",
                competition,
                len(candidates),
            )
            return None

        return CompetitionMetadata(
            slug=competition,
            title=getattr(match, "title", "") or "",
            description=getattr(match, "description", "") or "",
            category=getattr(match, "category", "") or "",
            evaluation_metric_raw=getattr(match, "evaluation_metric", "") or "",
            deadline=self._format_deadline(getattr(match, "deadline", None)),
            max_daily_submissions=self._coerce_int(
                getattr(match, "max_daily_submissions", None)
                or getattr(match, "maxDailySubmissions", None)
            ),
            submissions_disabled=bool(getattr(match, "submissions_disabled", False)),
            is_kernels_submissions_only=bool(
                getattr(match, "is_kernels_submissions_only", False)
            ),
            tags=self._extract_tags(match),
        )

    def count_todays_submissions(self, competition: str) -> int:
        """Count submissions made today for quota pre-flight checks."""
        api = self.authenticate()
        try:
            submissions = api.competition_submissions(competition)
        except Exception:
            logger.warning("Unable to count today's submissions for '%s'.", competition)
            return 0

        today = date.today()
        count = 0
        for entry in submissions or []:
            submitted = getattr(entry, "date", None) or getattr(entry, "submitted_at", None)
            if submitted is None:
                continue
            if isinstance(submitted, datetime):
                submitted_date = submitted.date()
            else:
                try:
                    submitted_date = datetime.fromisoformat(str(submitted).replace("Z", "+00:00")).date()
                except ValueError:
                    continue
            if submitted_date == today:
                count += 1
        return count

    @staticmethod
    def _format_deadline(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_tags(match: Any) -> list[str]:
        tags = getattr(match, "tags", None) or []
        result: list[str] = []
        for tag in tags:
            name = getattr(tag, "name", None) or getattr(tag, "ref", None) or str(tag)
            if name:
                result.append(str(name))
        category = getattr(match, "category", "") or ""
        if category and category not in result:
            result.insert(0, category)
        return result

    @staticmethod
    def _ref_slug(ref: str) -> str:
        """`ref` is a full competition URL (e.g. ".../competitions/titanic"),
        not a bare slug, so pull the slug back out to compare against
        the `--competition` string the pipeline was invoked with."""
        return ref.strip().rstrip("/").rsplit("/", 1)[-1].lower()

    @staticmethod
    def save_result(run_dir: Path, result: SubmissionResult) -> Path:
        output = run_dir / "submission_result.json"
        output.write_text(result.model_dump_json(indent=2))
        return output
