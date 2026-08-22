"""M24: a competition's file list, without its bytes.

`capture` walks a local directory, so a media competition costs its full
download before it can be a fixture. That is why the corpus is five tabular
fixtures and none of the text, image or audio ones the plan names —
`biohub-cell-tracking` alone is ≥0.99 GB in its first two hundred files.

The listing is what modality detection actually reads: `_detect_image` counts by
extension and never opens a file. These tests drive a fake API, because the
point is the paging, the refusal and the provenance, not the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.accessor.benchmark.capture import capture_from_listing
from labpilot.accessor.benchmark.expand import expand_fixture
from labpilot.accessor.benchmark.remote import (
    ListingUnavailable,
    RemoteFile,
    RemoteListing,
    fetch_listing,
)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch) -> None:
    """The pauses are real seconds, and this file exercises the retry path.

    Left alone, the rate-limit test alone spends ~17s in backoff — a unit suite
    that waits is a unit suite people stop running. The *durations* are not what
    these tests are about; that pages are retried and eventually refused is.
    """
    import labpilot.accessor.benchmark.remote as remote

    monkeypatch.setattr(remote, "sleep", lambda _seconds: None)


class _Entry:
    def __init__(self, name: str, size: int) -> None:
        self.name, self.total_bytes = name, size


class _Page:
    def __init__(self, files: list[_Entry], token: str | None) -> None:
        self.files, self.next_page_token = files, token


class _Api:
    """Pages through a fixed list, and can be told to rate-limit."""

    def __init__(self, names: list[str], *, limit_after: int | None = None) -> None:
        self.names = names
        self.limit_after = limit_after
        self.calls = 0

    def competition_list_files(self, competition, page_token=None, page_size=200):
        self.calls += 1
        if self.limit_after is not None and self.calls > self.limit_after:
            raise RuntimeError("429 Client Error: Too Many Requests")
        start = int(page_token or 0)
        chunk = self.names[start : start + page_size]
        nxt = str(start + page_size) if start + page_size < len(self.names) else None
        return _Page([_Entry(n, 100) for n in chunk], nxt)


# --- paging ---------------------------------------------------------------------


def test_a_listing_pages_to_the_end() -> None:
    """`biohub` is ≥23,800 files. One page of two hundred describes a different
    dataset from the one that exists."""
    api = _Api([f"train/img_{i:05}.png" for i in range(450)])

    listing = fetch_listing("comp", api)

    assert len(listing.files) == 450
    assert listing.complete
    assert api.calls == 3


def test_a_truncated_listing_is_refused_rather_than_returned() -> None:
    """Counts and ratios are the fixture's whole content, and half of them
    describe a dataset that does not exist.

    `_detect_image` decides on ratios — a partial listing does not fail to
    validate modality detection, it validates the wrong answer.
    """
    api = _Api([f"f{i}.png" for i in range(1000)], limit_after=1)

    with pytest.raises(ListingUnavailable, match="rate-limited"):
        fetch_listing("comp", api)


def test_a_listing_that_will_not_end_is_refused_too(monkeypatch) -> None:
    """The page cap is the other way a listing can come back short.

    The rate-limit test covers one route to a truncated listing and left this
    one uncovered — a mutation returning `complete=False` instead of raising
    passed the whole file. Both routes must refuse, because a fixture built from
    either has the wrong ratios.
    """
    import labpilot.accessor.benchmark.remote as remote

    monkeypatch.setattr(remote, "_MAX_PAGES", 2)
    api = _Api([f"f{i}.png" for i in range(1000)])

    with pytest.raises(ListingUnavailable, match="did not finish listing"):
        fetch_listing("comp", api)


def test_a_rate_limit_is_retried_before_it_is_reported() -> None:
    """Being told to "wait and re-run" restarts a hundred-odd requests and hits
    the same limit at the same place. Backoff belongs in the client."""

    class _Flaky(_Api):
        def competition_list_files(self, competition, page_token=None, page_size=200):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("429 Too Many Requests")
            start = int(page_token or 0)
            chunk = self.names[start : start + page_size]
            nxt = str(start + page_size) if start + page_size < len(self.names) else None
            return _Page([_Entry(n, 100) for n in chunk], nxt)

    listing = fetch_listing("comp", _Flaky([f"f{i}.png" for i in range(300)]))

    assert listing.complete and len(listing.files) == 300


def test_a_non_rate_limit_error_is_not_retried() -> None:
    """A missing competition does not become available by waiting."""

    class _Missing(_Api):
        def competition_list_files(self, *a, **k):
            raise RuntimeError("404 Not Found")

    with pytest.raises(ListingUnavailable, match="could not list"):
        fetch_listing("nope", _Missing([]))


# --- capture from a listing --------------------------------------------------------


def test_an_incomplete_listing_is_refused(tmp_path: Path) -> None:
    """Review finding: the field was declared and never checked.

    `fetch_listing` refuses to *return* a partial listing because its counts and
    ratios are the fixture's entire content. This path built one anyway, twenty
    lines further down the same module — the guard and the violation were in one
    file, written the same afternoon.
    """
    partial = RemoteListing(slug="c", files=(RemoteFile("a.png", 1),), complete=False)

    with pytest.raises(ValueError, match="incomplete listing"):
        capture_from_listing(partial, tmp_path, slug="c", fetch=lambda n, t: None)


def test_the_whole_listing_is_written_or_none_of_it(tmp_path: Path) -> None:
    """Review finding: a cap truncated the listing and the fixture reported the
    pre-truncation count.

    A provenance record that misstates its own coverage is worse than one that
    omits it — `tests/fixtures/real_failures/MANIFEST.md` exists because a
    79-byte fragment once claimed to be 624 bytes. `biohub` is ≥23,800 files, so
    the old 50,000 default was a cap already measured as reachable.
    """

    def fetch(name: str, target: Path) -> None:  # pragma: no cover - no tables here
        raise AssertionError("nothing tabular in this listing")

    names = [f"img_{i:05}.png" for i in range(1200)]
    fixture = capture_from_listing(_listing(names), tmp_path, slug="c", fetch=fetch)

    rows = (tmp_path / "listing.tsv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1200
    assert "1200 file(s)" in fixture.unverifiable["byte_fidelity"], "the count must match"


def test_a_name_that_escapes_the_destination_is_refused(tmp_path: Path) -> None:
    """Review finding. `capture` opens with "read-only by construction", and a
    tool that writes outside the directory it was pointed at has broken that
    whatever supplied the name.

    Refused and *recorded*: a file the capture would not take is not a file that
    does not exist, and a quietly short listing has the wrong ratios in exactly
    the way this path is built to avoid.
    """
    destination = tmp_path / "dest"
    listing = RemoteListing(
        slug="c",
        files=(
            # Out of the capture entirely.
            RemoteFile("../../escaped.csv", 10),
            # And the nastier one: only out of `data/`, which lands on the
            # manifest. Checking containment against the destination rather than
            # `data/` lets this through, and the fixture's own record of what it
            # listed is what gets overwritten.
            RemoteFile("../fixture.json", 10),
            RemoteFile("ok.png", 10),
        ),
        complete=True,
    )

    fixture = capture_from_listing(
        listing, destination, slug="c", fetch=lambda n, t: t.write_text("id\n", encoding="utf-8")
    )

    assert not (tmp_path / "escaped.csv").exists()
    assert not (tmp_path.parent / "escaped.csv").exists()
    refused = fixture.unverifiable["listing_completeness"]
    assert "escaped.csv" in refused and "fixture.json" in refused
    assert "ok.png" in (destination / "listing.tsv").read_text(encoding="utf-8")
    assert fixture.slug == "c", "the manifest survived"


def test_the_cli_is_the_production_caller() -> None:
    """Review finding: neither entry point had one, so the feature was reachable
    only from tests — the same gap #164 and #165 needed noted after review."""
    import inspect

    from labpilot.cli import bench_cli

    source = inspect.getsource(bench_cli)

    assert "fetch_listing(slug, api)" in source
    assert "capture_from_listing(" in source


def _listing(names: list[str]) -> RemoteListing:
    return RemoteListing(
        slug="comp", files=tuple(RemoteFile(n, 4_500_000) for n in names), complete=True
    )


def test_only_tabular_files_are_fetched(tmp_path: Path) -> None:
    """The whole point: a media competition becomes a fixture without its bytes.

    A header is the one thing a listing cannot carry and five criteria depend on
    it, so CSVs are fetched and nothing else is.
    """
    fetched: list[str] = []

    def fetch(name: str, target: Path) -> None:
        fetched.append(name)
        target.write_text("id,label\n1,cat\n", encoding="utf-8")

    names = ["sample_submission.csv"] + [f"test/vol_{i}.zarr/0/c/{i}" for i in range(200)]
    capture_from_listing(_listing(names), tmp_path, slug="comp", fetch=fetch)

    assert fetched == ["sample_submission.csv"], "200 zarr chunks stayed where they were"
    assert (tmp_path / "listing.tsv").read_text(encoding="utf-8").count("\n") == 200


def test_the_fixture_says_its_listing_carries_no_checksums(tmp_path: Path) -> None:
    """The API returns none, and an empty sha column is one character from a sha
    of nothing. `listing_source` is what stops a reader assuming."""

    def fetch(name: str, target: Path) -> None:
        target.write_text("id,label\n1,cat\n", encoding="utf-8")

    fixture = capture_from_listing(
        _listing(["train.csv", "img/a.png"]), tmp_path, slug="comp", fetch=fetch
    )

    assert fixture.listing_source == "remote"
    assert "no checksum" in fixture.unverifiable["byte_fidelity"]


def test_a_captured_header_still_records_the_real_file(tmp_path: Path) -> None:
    """`source_sha256` and `source_bytes` are of the downloaded file, so a
    re-download can be checked against them even though only the header stays."""

    def fetch(name: str, target: Path) -> None:
        target.write_text("id,label\n1,cat\n2,dog\n", encoding="utf-8")

    fixture = capture_from_listing(_listing(["train.csv"]), tmp_path, slug="comp", fetch=fetch)

    entry = next(f for f in fixture.files if f.path == "train.csv")
    assert entry.source_rows == 2 and entry.fixture_rows == 0
    assert len(entry.source_sha256) == 64
    assert (tmp_path / "data/train.csv").read_text(encoding="utf-8") == "id,label\n"


def test_an_unfetchable_file_is_listed_rather_than_lost(tmp_path: Path) -> None:
    """One file that will not download is not a failed capture — it is a file
    the fixture does not carry, and the listing still records that it exists."""

    def fetch(name: str, target: Path) -> None:
        raise OSError("gone")

    fixture = capture_from_listing(_listing(["train.csv"]), tmp_path, slug="comp", fetch=fetch)

    assert fixture.files == []
    assert "train.csv" in (tmp_path / "listing.tsv").read_text(encoding="utf-8")


def test_the_expander_rebuilds_the_ratios(tmp_path: Path) -> None:
    """The reason a listing is enough. `_detect_image` counts by extension and
    never opens a file, so placeholders at the right paths carry the ratio that
    a proportionally shrunk tree would destroy."""

    def fetch(name: str, target: Path) -> None:
        target.write_text("id,label\n1,cat\n", encoding="utf-8")

    names = ["train.csv"] + [f"train/img_{i:04}.png" for i in range(500)]
    capture_from_listing(_listing(names), tmp_path / "fixture", slug="comp", fetch=fetch)

    expand_fixture(tmp_path / "fixture", tmp_path / "expanded")

    images = list((tmp_path / "expanded").rglob("*.png"))
    assert len(images) == 500, "the ratio modality detection decides on"
    assert all(p.stat().st_size == 0 for p in images)
    assert (tmp_path / "expanded/train.csv").read_text(encoding="utf-8") == "id,label\n"
