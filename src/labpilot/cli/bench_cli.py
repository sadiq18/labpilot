"""``research bench`` — capture competitions, and score the system against them.

The operator's side of M24. `capture` reads a real dataset **read-only** and
writes a fixture; `score` runs the shipped path against every fixture and prints
the table the milestone's success criteria are read from.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.accessor.benchmark.capture import capture_competition
from labpilot.accessor.benchmark.fixture import load_fixture
from labpilot.accessor.benchmark.ledger import corpus_hash, load_ledger
from labpilot.accessor.benchmark.score import CRITERIA, profile_and_score

bench_app = typer.Typer(
    help="Capture competitions into the corpus, and score understanding against it.",
    no_args_is_help=True,
)
console = Console()

CORPUS = Path("tests/fixtures/competitions")

_VERDICT_STYLE = {
    "pass": "green",
    "fail": "red",
    "known_failure": "yellow",
    "unverifiable": "dim",
    "not_applicable": "dim",
}


@bench_app.command("capture-remote")
def capture_remote(
    slug: str = typer.Argument(..., help="Competition slug"),
    into: Path = typer.Option(CORPUS, "--into", help="Corpus directory"),
    licence: str = typer.Option("unknown", "--licence"),
    redistribution: str = typer.Option("unknown", "--redistribution"),
) -> None:
    """Capture a competition from its Kaggle file list, without its bytes.

    What this is for: a media competition costs its full download before it can
    be a fixture, which is why the corpus is five tabular ones.
    `biohub-cell-tracking` is 4.5 MB zarr chunks and ≥0.99 GB in its first two
    hundred files; its listing is a few hundred kilobytes, and the listing is
    what `_detect_image` reads — it counts by extension and never opens a file.

    Only tabular files are downloaded, because a header is the one thing a
    listing cannot carry.
    """
    from labpilot.accessor.benchmark.capture import capture_from_listing
    from labpilot.accessor.benchmark.remote import ListingUnavailable, fetch_listing
    from labpilot.accessor.kaggle.client import KaggleClient
    from labpilot.config import load_config

    client = KaggleClient(load_config().kaggle)
    try:
        api = client.authenticate()
    except Exception as exc:  # noqa: BLE001 — say which half failed
        console.print(f"[red]Kaggle authentication failed:[/red] {exc}")
        raise typer.Exit(2) from exc

    try:
        listing = fetch_listing(slug, api)
    except ListingUnavailable as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(
        f"[dim]{len(listing.files):,} file(s), {listing.total_bytes / 1e9:.2f} GB "
        f"in the real dataset[/dim]"
    )

    destination = Path(into) / slug
    fixture = capture_from_listing(
        listing,
        destination,
        slug=slug,
        fetch=lambda name, target: api.competition_download_file(
            slug, name, path=str(target.parent), quiet=True
        ),
        licence=licence,
        redistribution=redistribution,
    )
    console.print(
        f"[green]Captured[/green] {slug} -> {destination}\n"
        f"  {len(fixture.files)} table(s) by header, "
        f"{len(listing.files) - len(fixture.files)} file(s) by name and size"
    )
    console.print(
        "\n[dim]Now fill `expected` in fixture.json from the competition's own rules "
        "page — never from what the profiler produced.[/dim]"
    )


@bench_app.command("capture")
def capture(
    data: Path = typer.Argument(..., help="The competition's data directory"),
    slug: str = typer.Option(..., "--slug", help="Competition slug"),
    spec: Path = typer.Option(None, "--spec", help="Its competition.json"),
    mode: str = typer.Option(
        "headers_only", "--mode", help="headers_only | head:N | stride:K | verbatim"
    ),
    into: Path = typer.Option(CORPUS, "--into", help="Corpus directory"),
    max_per_directory: int = typer.Option(
        None, "--max-per-directory", help="Cap files taken from any one directory"
    ),
) -> None:
    """Capture one competition. The source is never written to."""
    destination = into / slug
    try:
        fixture = capture_competition(
            data,
            destination,
            slug=slug,
            mode=mode,
            spec_path=spec,
            max_per_directory=max_per_directory,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Captured[/green] {slug} → {destination} ({len(fixture.files)} file(s))")
    if fixture.unverifiable:
        console.print("Cannot be scored from this capture:")
        for criterion, reason in sorted(fixture.unverifiable.items()):
            console.print(f"  · {criterion} — {reason}")
    console.print(
        "\nNow fill `expected` in fixture.json from the competition's own rules page — "
        "never from what the profiler produced, which would score the system against itself."
    )


@bench_app.command("score")
def score(
    corpus: Path = typer.Option(CORPUS, "--corpus", help="Corpus directory"),
) -> None:
    """Score every captured competition, and print the aggregate."""
    slugs = sorted(p.name for p in corpus.iterdir() if (p / "fixture.json").is_file())
    if not slugs:
        console.print(f"[red]No fixtures under {corpus}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Competition corpus — {len(slugs)} fixture(s)")
    table.add_column("competition", overflow="fold")
    for criterion in CRITERIA:
        table.add_column(criterion.replace("_", " ")[:12])

    understood = 0
    per_criterion: dict[str, list[str]] = {criterion: [] for criterion in CRITERIA}
    with tempfile.TemporaryDirectory() as workroot:
        for slug in slugs:
            card = profile_and_score(corpus / slug, Path(workroot) / slug)
            understood += 1 if card.understood else 0
            cells = []
            for criterion in CRITERIA:
                verdict = card.verdict_for(criterion) or "—"
                per_criterion[criterion].append(verdict)
                style = _VERDICT_STYLE.get(verdict, "")
                cells.append(f"[{style}]{verdict[:4]}[/{style}]" if style else verdict[:4])
            table.add_row(slug, *cells)
    console.print(table)

    console.print(f"\nUnderstood (every applicable criterion passes): {understood}/{len(slugs)}")

    ledger = load_ledger(corpus)
    for criterion, verdicts in per_criterion.items():
        scored = [v for v in verdicts if v in ("pass", "fail", "known_failure")]
        if not scored:
            continue
        passed = sum(1 for v in scored if v == "pass")
        line = f"  {criterion:26} {passed}/{len(scored)} of the fixtures that can score it"
        # The floor beside the number, because a rate on its own says nothing
        # about whether it may drop. The gap to the goal is the point of the
        # ratchet: 0.95 asserted on day one makes the suite red and teaches
        # everyone to ignore it.
        floor = (ledger.floors if ledger else {}).get(criterion)
        if floor is not None:
            reached = passed / len(scored)
            mark = "" if reached >= floor else "  [red]below floor[/red]"
            goal = "" if reached >= (ledger.goal if ledger else 1.0) else "  [dim]< goal[/dim]"
            line += f"  [dim](floor {floor:.2f})[/dim]{mark}{goal}"
        console.print(line)

    if ledger is not None:
        stale = "" if ledger.corpus_hash == corpus_hash(corpus) else "  [yellow](stale)[/yellow]"
        console.print(
            f"\n[dim]corpus {corpus_hash(corpus)[:12]} · floors recorded "
            f"{ledger.recorded_at} · goal {ledger.goal:.2f}[/dim]{stale}"
        )


@bench_app.command("show")
def show(
    slug: str = typer.Argument(..., help="Competition slug"),
    corpus: Path = typer.Option(CORPUS, "--corpus", help="Corpus directory"),
) -> None:
    """What one fixture kept, lost, and expects."""
    fixture = load_fixture(corpus / slug)
    console.print(f"[bold]{fixture.slug}[/bold] — captured {fixture.captured_at}")
    console.print(f"provenance: {fixture.provenance} · licence: {fixture.licence}")
    for entry in fixture.files:
        rows = f"{entry.fixture_rows}/{entry.source_rows} rows" if entry.source_rows else "no rows"
        console.print(f"  {entry.path:44} {entry.mode:14} {rows}")
    for criterion, reason in sorted(fixture.unverifiable.items()):
        console.print(f"  [dim]unverifiable[/dim] {criterion} — {reason}")
    for criterion, reason in sorted(fixture.known_failures.items()):
        console.print(f"  [yellow]known failure[/yellow] {criterion} — {reason}")
