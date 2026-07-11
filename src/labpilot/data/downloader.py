from pathlib import Path

from labpilot.config import KaggleConfig


class DataLayout:
    """Standard directory layout for a run's data artifacts."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.raw_dir = run_dir / "data" / "raw"
        self.processed_dir = run_dir / "data" / "processed"

    def ensure(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def list_raw_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        return sorted(p for p in self.raw_dir.rglob("*") if p.is_file())


class DataDownloader:
    """Download competition data via the Kaggle API."""

    def __init__(self, competition_slug: str, config: KaggleConfig) -> None:
        self.competition_slug = competition_slug
        self.config = config

    def download(self, run_dir: Path) -> Path:
        layout = DataLayout(run_dir)
        layout.ensure()

        # TODO: call kaggle.competition_download_files and unzip
        readme = layout.raw_dir / "README.txt"
        readme.write_text(
            f"Placeholder for {self.competition_slug} data.\n"
            "Implement Kaggle API download in DataDownloader.download().\n"
        )
        return layout.raw_dir
