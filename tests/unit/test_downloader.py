from pathlib import Path

from labpilot.config import KaggleConfig
from labpilot.data.downloader import DataDownloader


class CountingGateway:
    """Fake Kaggle gateway that records how many times it was asked to download."""

    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.download_calls = 0

    def download_competition(self, competition: str, destination: Path) -> list[Path]:
        self.download_calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, content in self.files.items():
            path = destination / name
            path.write_text(content)
            paths.append(path)
        return sorted(paths)

    def upload_submission(self, *args, **kwargs):
        raise NotImplementedError


def test_download_uses_cache_on_second_call(tmp_path: Path):
    gateway = CountingGateway({"train.csv": "id,target\n1,0\n"})
    config = KaggleConfig(cache_dir=tmp_path / "cache")

    first_run = tmp_path / "run-1"
    second_run = tmp_path / "run-2"

    downloader = DataDownloader("generic-competition", config, client=gateway)
    downloader.download(first_run)
    downloader.download(second_run)

    assert gateway.download_calls == 1
    assert (first_run / "data" / "raw" / "train.csv").exists()
    assert (second_run / "data" / "raw" / "train.csv").exists()


def test_download_populates_cache_per_competition(tmp_path: Path):
    gateway = CountingGateway({"train.csv": "id,target\n1,0\n"})
    config = KaggleConfig(cache_dir=tmp_path / "cache")

    downloader = DataDownloader("generic-competition", config, client=gateway)
    downloader.download(tmp_path / "run")

    cache_dir = config.cache_dir / "generic-competition"
    assert (cache_dir / "train.csv").exists()
