from pathlib import Path

from labpilot.config import KaggleConfig
from labpilot.kaggle.client import KaggleClient, KaggleGateway


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

    def __init__(
        self,
        competition_slug: str,
        config: KaggleConfig,
        client: KaggleGateway | None = None,
    ) -> None:
        self.competition_slug = competition_slug
        self.config = config
        self.client = client or KaggleClient(config)

    def download(self, run_dir: Path) -> Path:
        layout = DataLayout(run_dir)
        layout.ensure()

        self.client.download_competition(self.competition_slug, layout.raw_dir)
        if not layout.list_raw_files():
            raise RuntimeError(f"No data files downloaded for {self.competition_slug}.")
        return layout.raw_dir
