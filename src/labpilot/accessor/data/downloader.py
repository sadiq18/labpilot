import logging
import shutil
import zipfile
from pathlib import Path

from labpilot.config import KaggleConfig
from labpilot.accessor.kaggle.client import KaggleClient, KaggleGateway

logger = logging.getLogger(__name__)


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
        return list_files(self.raw_dir)


def list_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file())


class DataDownloader:
    """Download competition data via the Kaggle API.

    Downloads are cached on disk per competition slug (`KaggleConfig.cache_dir`)
    so that re-running the same competition across multiple research runs does
    not re-download identical data from Kaggle every time.
    """

    def __init__(
        self,
        competition_slug: str,
        config: KaggleConfig,
        client: KaggleGateway | None = None,
    ) -> None:
        self.competition_slug = competition_slug
        self.config = config
        self.client = client or KaggleClient(config)

    @property
    def cache_dir(self) -> Path:
        return self.config.cache_dir / self.competition_slug

    def download(self, run_dir: Path) -> Path:
        layout = DataLayout(run_dir)
        layout.ensure()

        cache_dir = self.cache_dir
        cached_files = list_files(cache_dir)
        if cached_files:
            logger.info(
                "Reusing cached data for '%s' from %s (%d files); skipping download.",
                self.competition_slug,
                cache_dir,
                len(cached_files),
            )
        else:
            logger.info(
                "Downloading data for '%s' into cache %s.", self.competition_slug, cache_dir
            )
            self.client.download_competition(self.competition_slug, cache_dir)
            if not list_files(cache_dir):
                raise RuntimeError(f"No data files downloaded for {self.competition_slug}.")

        self._extract_zip_archives(cache_dir)
        self._sync_from_cache(cache_dir, layout.raw_dir)
        self._extract_zip_archives(layout.raw_dir)
        if not layout.list_raw_files():
            raise RuntimeError(f"No data files available for {self.competition_slug}.")
        return layout.raw_dir

    @staticmethod
    def _extract_zip_archives(directory: Path, *, max_passes: int = 10) -> None:
        for _ in range(max_passes):
            archives = sorted(directory.glob("*.zip"))
            if not archives:
                return
            for archive in archives:
                logger.info("Extracting %s", archive)
                with zipfile.ZipFile(archive) as zipped:
                    zipped.extractall(directory)
                archive.unlink(missing_ok=True)

    def _sync_from_cache(self, cache_dir: Path, raw_dir: Path) -> None:
        for cached_file in list_files(cache_dir):
            relative = cached_file.relative_to(cache_dir)
            destination = raw_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(cached_file, destination)
