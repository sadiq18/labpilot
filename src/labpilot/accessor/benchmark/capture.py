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
from datetime import UTC, datetime
from pathlib import Path

from labpilot.accessor.benchmark.fixture import (
    CapturedFile,
    CompetitionFixture,
    Expectations,
    save_fixture,
)

__all__ = ["capture_competition", "parse_mode"]

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


def _digest(path: Path) -> tuple[str, int, int | None]:
    """sha256, byte count, and line count of the source file."""
    digest = hashlib.sha256()
    size = 0
    lines = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
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
    seen_per_directory: dict[str, int] = {}
    sources = sorted(source.rglob("*"))
    if spec_path is not None and Path(spec_path).is_file():
        sources.append(Path(spec_path))
    for path in sources:
        if not path.is_file():
            continue
        relative = (
            path.name
            if spec_path is not None and path == Path(spec_path)
            else str(path.relative_to(source))
        )
        parent = str(Path(relative).parent)
        if max_per_directory is not None:
            seen_per_directory[parent] = seen_per_directory.get(parent, 0) + 1
            if seen_per_directory[parent] > max_per_directory:
                continue
        sha, size, lines = _digest(path)
        if path.suffix.lower() in _TABULAR_SUFFIXES or path.name in _VERBATIM_NAMES:
            file_kind = "verbatim" if path.name in _VERBATIM_NAMES else kind
            rows = _rows_for(path, file_kind, n)
            target = destination / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(rows), encoding="utf-8")
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
    if kind == "headers_only":
        unverifiable["feature_columns"] = (
            "no rows, so constant and equals-target exclusions cannot fire"
        )

    fixture = CompetitionFixture(
        slug=slug,
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
