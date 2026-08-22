"""Turn a real competition into a fixture, without touching it.

M24. Read-only by construction: nothing here opens a source file for writing,
which is the same rule AGENTS.md states for competition workspaces — *"if a
workspace needs migrating or cleaning, make labpilot do it on the next run; do
not hand-edit artifacts"*.

The output is a directory the harness can expand and score, plus a manifest that
says what was kept and what was lost. A fixture that does not say what it lost
is a paraphrase.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from labpilot.accessor.benchmark.fixture import (
    CapturedFile,
    CompetitionFixture,
    Expectations,
    save_fixture,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from labpilot.accessor.benchmark.remote import RemoteListing

logger = logging.getLogger(__name__)

__all__ = ["capture_competition", "capture_from_listing", "parse_mode"]

#: Files worth carrying whole: they are metadata, not data, and the metric
#: criterion cannot be scored without them.
_VERBATIM_NAMES = ("competition.json",)
#: What a capture will read as tabular. Anything else is listed, never copied.
_TABULAR_SUFFIXES = (".csv", ".tsv")


def parse_mode(mode: str) -> tuple[str, int]:
    """``"stride:25"`` → ``("stride", 25)``. Raises on anything else.

    A typo'd mode that silently fell back to a default would produce a fixture
    whose manifest describes a capture that did not happen.
    """
    kind, _, raw = mode.partition(":")
    if kind in ("verbatim", "headers_only"):
        if raw:
            raise ValueError(f"{kind!r} takes no argument; got {mode!r}")
        return kind, 0
    if kind in ("head", "stride"):
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{kind!r} needs a positive integer, as in {kind}:25; got {mode!r}")
        return kind, int(raw)
    raise ValueError(f"unknown capture mode {mode!r}")


def _digest(path: Path) -> tuple[str, int, int]:
    """sha256, byte count, and line count of the source file.

    A file whose last line has no terminator still has that line. Counting
    ``\n`` alone lost it, which made `source_rows` one short — and a capture of
    such a file recorded more `fixture_rows` than the source it came from, which
    is the manifest claiming the fixture holds rows the dataset does not.
    """
    digest = hashlib.sha256()
    size = 0
    lines = 0
    last = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if size and last != b"\n":
        lines += 1
    return digest.hexdigest(), size, lines


def _rows_for(path: Path, kind: str, n: int) -> list[str]:
    """The lines a mode keeps, header first."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        if not header:
            return []
        if kind == "headers_only":
            return [header]
        if kind == "verbatim":
            return [header, *handle.readlines()]
        if kind == "head":
            kept = []
            for _ in range(n):
                line = handle.readline()
                if not line:
                    break
                kept.append(line)
            return [header, *kept]
        # stride: every nth row, which is what preserves a contiguous prefix and
        # a contiguous suffix through a truncation that `head` would destroy.
        return [header, *[line for index, line in enumerate(handle) if index % n == 0]]


def capture_from_listing(
    listing: RemoteListing,
    destination: Path,
    *,
    slug: str,
    fetch: Callable[[str, Path], None],
    expected: Expectations | None = None,
    licence: str = "unknown",
    redistribution: str = "unknown",
    max_listed: int = 50_000,
) -> CompetitionFixture:
    """Capture from a file list, downloading only what has to be read.

    The move this makes possible: a media competition becomes a fixture without
    its bytes. `biohub-cell-tracking` is 4.5 MB zarr chunks and ≥0.99 GB in its
    first two hundred files; its *listing* is a few hundred kilobytes of text,
    and the listing is all `_detect_image` needs — it counts by extension and
    never opens a file.

    Only tabular files are fetched, because a header is the one thing a listing
    cannot carry and five criteria depend on it. Everything else is recorded at
    name and size, which is what the expander materialises as placeholders.

    `fetch` is injected rather than imported: this module may not depend on a
    Kaggle client, and a test must be able to capture without a network.
    """
    destination = Path(destination)
    (destination / "data").mkdir(parents=True, exist_ok=True)

    captured: list[CapturedFile] = []
    listed: list[str] = []
    for entry in listing.files:
        if Path(entry.name).suffix.lower() not in _TABULAR_SUFFIXES:
            # No sha: the API returns none, and an empty column is not a hash of
            # nothing. `listing_source="remote"` on the fixture is what says so.
            listed.append(f"{entry.name}\t{entry.size}\t")
            continue
        target = destination / "data" / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch(entry.name, target)
        except Exception as exc:  # noqa: BLE001 — one unfetchable file is not a
            # failed capture; it is a file the fixture does not carry, and the
            # listing still records that it exists.
            logger.info("Could not fetch %s from %s: %s", entry.name, slug, exc)
            listed.append(f"{entry.name}\t{entry.size}\t")
            continue
        sha, size, lines = _digest(target)
        header = _rows_for(target, "headers_only", 0)
        target.write_text("".join(header), encoding="utf-8")
        captured.append(
            CapturedFile(
                path=entry.name,
                mode="headers_only",
                # Of the *real* file, so a re-download can be checked against it
                # even though only its header survives here.
                source_sha256=sha,
                source_bytes=size,
                source_rows=max(lines - 1, 0),
                fixture_rows=0,
            )
        )

    if listed:
        (destination / "listing.tsv").write_text(
            "\n".join(listed[:max_listed]) + "\n", encoding="utf-8"
        )

    unverifiable = {
        "row_count": "headers_only capture keeps a subset of rows",
        "cardinality": "headers_only capture keeps a subset of rows",
        "feature_columns": "no rows, so constant and equals-target exclusions cannot fire",
    }
    if listed:
        unverifiable["byte_fidelity"] = (
            f"{len(listed)} file(s) are recorded by name and size from the Kaggle API, "
            "which returns no checksum — the listing proves they exist, not what they hold"
        )

    fixture = CompetitionFixture(
        slug=slug,
        captured_at=datetime.now(UTC).date().isoformat(),
        source=f"kaggle api: {slug}",
        provenance="derived",
        licence=licence,
        redistribution=redistribution,  # type: ignore[arg-type]
        listing_source="remote" if listed else "none",
        files=captured,
        expected=expected or Expectations(),
        unverifiable=unverifiable,
    )
    save_fixture(destination, fixture)
    return fixture


def capture_competition(
    source: Path,
    destination: Path,
    *,
    slug: str,
    mode: str = "headers_only",
    expected: Expectations | None = None,
    licence: str = "unknown",
    redistribution: str = "unknown",
    max_listed: int = 5000,
    spec_path: Path | None = None,
    max_per_directory: int | None = None,
) -> CompetitionFixture:
    """Capture `source` into `destination`, and return the manifest written.

    Non-tabular files are *listed*, never copied: for media the facts live in
    counts and ratios, and a proportionally shrunk tree does not fail to
    validate modality detection — it validates the wrong answer.

    `spec_path` points at the competition's `competition.json`, which lives at
    the workspace root rather than inside the data directory — and without it
    the metric criterion cannot be scored at all.

    `max_per_directory` caps how many files are taken from any one directory. A
    per-entity dataset can hold 1,546 tables whose *structure* — two kinds, the
    columns each withholds — is carried by a handful. It changes counts, which
    every capture mode already declares unverifiable, and it is recorded so the
    fixture never claims to be the whole thing.
    """
    kind, n = parse_mode(mode)
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(f"no dataset at {source}")

    captured: list[CapturedFile] = []
    listing: list[str] = []
    lossy: list[str] = []
    seen_per_directory: dict[str, int] = {}
    sources = sorted(source.rglob("*"))
    spec = Path(spec_path).resolve() if spec_path is not None else None
    # Only when it is not already in the tree. A spec that lives inside the data
    # directory was walked above, and appending it again put two entries in the
    # manifest for one file — invisible to a check that compares *sets*, which
    # is the provenance defect this corpus exists to prevent.
    if spec is not None and spec.is_file() and spec not in {p.resolve() for p in sources}:
        sources.append(spec)
    for path in sources:
        if not path.is_file():
            continue
        is_spec = spec is not None and path.resolve() == spec
        relative = path.name if is_spec else str(path.relative_to(source))
        sha, size, lines = _digest(path)
        if path.suffix.lower() in _TABULAR_SUFFIXES or path.name in _VERBATIM_NAMES:
            # The cap counts what is *captured*. Counting every file meant six
            # images ahead of `train.csv` alphabetically exhausted the budget
            # and the fixture captured no tables at all — which is the layout of
            # every image competition in the corpus.
            parent = str(Path(relative).parent)
            if max_per_directory is not None and not is_spec:
                seen_per_directory[parent] = seen_per_directory.get(parent, 0) + 1
                if seen_per_directory[parent] > max_per_directory:
                    continue
            file_kind = "verbatim" if path.name in _VERBATIM_NAMES else kind
            target = destination / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if file_kind == "verbatim":
                # Byte for byte. Reading with `errors="replace"` and writing
                # UTF-8 back turned every latin-1 byte into U+FFFD while the
                # fixture still claimed `provenance: verbatim` — and the sha256
                # recorded is of the *source*, so the mismatch would have read
                # as "the dataset changed" rather than "the capture is lossy".
                shutil.copyfile(path, target)
                rows = _rows_for(path, "verbatim", n)
            else:
                rows = _rows_for(path, file_kind, n)
                text = "".join(rows)
                if "\ufffd" in text:
                    lossy.append(relative)
                target.write_text(text, encoding="utf-8")
            captured.append(
                CapturedFile(
                    path=relative,
                    mode=file_kind if file_kind in ("verbatim", "headers_only") else mode,
                    source_sha256=sha,
                    source_bytes=size,
                    # Rows are a tabular idea. A captured `competition.json` is
                    # carried whole and has none, and reporting its line count
                    # as a row count would be a number that means nothing.
                    source_rows=max(lines - 1, 0)
                    if path.suffix.lower() in _TABULAR_SUFFIXES
                    else None,
                    fixture_rows=max(len(rows) - 1, 0)
                    if path.suffix.lower() in _TABULAR_SUFFIXES
                    else None,
                )
            )
        else:
            # Listed with its size, so the expander can rebuild the tree — and
            # the ratios `_detect_image` decides on — without the bytes.
            listing.append(f"{relative}\t{size}\t{sha}")

    if listing:
        (destination / "listing.tsv").write_text(
            "\n".join(listing[:max_listed]) + "\n", encoding="utf-8"
        )

    unverifiable: dict[str, str] = {}
    if kind != "verbatim":
        unverifiable["row_count"] = f"{mode} capture keeps a subset of rows"
        unverifiable["cardinality"] = f"{mode} capture keeps a subset of rows"
    if max_per_directory is not None:
        unverifiable["partition_counts"] = (
            f"at most {max_per_directory} file(s) captured per directory"
        )
    if lossy:
        unverifiable["byte_fidelity"] = (
            f"{len(lossy)} file(s) held bytes that are not UTF-8 and were captured with "
            f"replacement characters: {', '.join(sorted(lossy)[:3])}"
        )
    if kind == "headers_only":
        unverifiable["feature_columns"] = (
            "no rows, so constant and equals-target exclusions cannot fire"
        )

    fixture = CompetitionFixture(
        slug=slug,
        listing_source="walked" if listing else "none",
        captured_at=datetime.now(UTC).date().isoformat(),
        source=str(source),
        provenance="verbatim" if kind == "verbatim" else "derived",
        licence=licence,
        redistribution=redistribution,  # type: ignore[arg-type]
        files=captured,
        expected=expected or Expectations(),
        unverifiable=unverifiable,
    )
    save_fixture(destination, fixture)
    return fixture
