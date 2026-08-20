"""Rebuild the shape a fixture was captured from.

M24. The key move, and the reason a hermetic corpus can stand in for real data:
`_detect_image` counts by extension and **never opens a file**, `_role_of` reads
directory parts, and `_match_filename_column` opens only the CSV. All of that is
satisfied by empty files at the right paths — thirty thousand of them in well
under a second — so the ratios modality detection decides on survive a capture
that carries none of the bytes.

Where decoding actually happens, the fixture ships the real bytes and the
expander overlays them. Nothing here invents a file that the listing did not
record.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from labpilot.accessor.benchmark.fixture import CompetitionFixture, load_fixture

__all__ = ["expand_fixture"]


def expand_fixture(directory: Path, destination: Path) -> CompetitionFixture:
    """Materialise `directory`'s fixture into `destination`, and return it.

    Tabular files are copied as captured. Everything in `listing.tsv` becomes a
    zero-byte placeholder at its recorded path — enough for every rule that
    counts, groups or names files, and nothing at all for a rule that reads one,
    which is the honest boundary.
    """
    directory = Path(directory)
    destination = Path(destination)
    fixture = load_fixture(directory)

    data = directory / "data"
    if data.is_dir():
        shutil.copytree(data, destination, dirs_exist_ok=True)

    listing = directory / "listing.tsv"
    if listing.is_file():
        for line in listing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            relative = line.split("\t", 1)[0]
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(b"")
    return fixture
