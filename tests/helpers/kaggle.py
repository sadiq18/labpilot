import shutil
from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionMetadata
from labpilot.accessor.kaggle.client import SubmissionResult


class FakeKaggleGateway:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.uploads: list[Path] = []

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        if any(self.source.rglob("*")):
            for source_file in self.source.rglob("*"):
                if source_file.is_file():
                    rel = source_file.relative_to(self.source)
                    destination_file = destination / rel
                    destination_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination_file)
                    files.append(destination_file)
        else:
            for source_file in self.source.glob("*.csv"):
                destination_file = destination / source_file.name
                shutil.copy2(source_file, destination_file)
                files.append(destination_file)
        return sorted(files)

    def upload_submission(
        self,
        competition: str,
        submission_path: Path,
        message: str | None = None,
    ) -> SubmissionResult:
        self.uploads.append(submission_path)
        return SubmissionResult(
            competition=competition,
            submission_path=str(submission_path),
            status="submitted",
            message=message or "test submission",
        )

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        return None
