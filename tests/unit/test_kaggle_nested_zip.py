import zipfile
from pathlib import Path

from labpilot.accessor.kaggle.client import KaggleClient


def test_extract_all_zip_archives_handles_nested_archives(tmp_path: Path):
    inner = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("train.csv", "id,target\n1,0\n")

    outer = tmp_path / "bundle.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.write(inner, arcname="inner.zip")

    inner.unlink()

    KaggleClient._extract_all_zip_archives(tmp_path)

    assert (tmp_path / "train.csv").is_file()
    assert not list(tmp_path.glob("*.zip"))
