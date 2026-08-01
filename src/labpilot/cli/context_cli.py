"""``research context`` — trust/debug CLI for the Context Engine.

Builds a real ``ContextBundle`` (retrieve → rank → compress), not the Plan 9
RI ``ResearchContext``. Offline-safe: no LLM required.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.research_engine.context import ContextBundle, ContextRequest, build_context

context_app = typer.Typer(
    help="Build and explain Context Engine bundles (trust/debug).",
    no_args_is_help=True,
)
console = Console()


def _build_bundle(
    *,
    competition: str | None,
    query: str,
    goal: str,
    max_items: int,
    max_chars: int,
    config_path: Path,
    knowledge_dir: Path | None,
    runs_dir: Path | None,
) -> ContextBundle:
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    slug = resolve_competition(competition, workspace)
    q = (query or goal or "").strip()
    request = ContextRequest(
        competition=slug,
        goal=goal or q,
        query=q,
        knowledge_dir=config.knowledge_dir,
        max_items=max_items,
        max_chars=max_chars,
    )
    return build_context(request)


def _print_summary(bundle: ContextBundle) -> None:
    console.print(
        f"\n[bold]Context bundle[/bold] — [cyan]{bundle.request.competition}[/cyan]"
    )
    console.print(
        f"[dim]query=[/dim]{bundle.request.query or '—'}  "
        f"[dim]items={len(bundle.items)}[/dim]  "
        f"[dim]built_at={bundle.built_at}[/dim]"
    )
    if bundle.notes:
        console.print("\n[bold]Pipeline notes[/bold]")
        for note in bundle.notes:
            console.print(f"  [yellow]•[/yellow] {note}")
    if bundle.provider_errors:
        console.print("\n[bold red]Provider errors[/bold red]")
        for err in bundle.provider_errors:
            console.print(f"  [red]•[/red] {err}")
    text = bundle.summary(max_chars=4000)
    console.print("\n[bold]Summary[/bold]")
    console.print(text or "(empty)")


def _print_explain(bundle: ContextBundle) -> None:
    console.print(
        f"\n[bold]Context explain[/bold] — [cyan]{bundle.request.competition}[/cyan]"
    )
    console.print(
        f"[dim]query=[/dim]{bundle.request.query or '—'}  "
        f"[dim]goal=[/dim]{bundle.request.goal or '—'}"
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("score", justify="right", width=7)
    table.add_column("source", width=12)
    table.add_column("kind", width=10)
    table.add_column("reason")
    table.add_column("text")
    for i, item in enumerate(bundle.items, start=1):
        snippet = (item.text or "").replace("\n", " ").strip()
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        table.add_row(
            str(i),
            f"{item.score:.3f}",
            item.source,
            item.kind,
            (item.reason or "")[:120],
            snippet,
        )
    if bundle.items:
        console.print(table)
    else:
        console.print("[dim](no items kept after retrieve → rank → compress)[/dim]")

    console.print("\n[bold]Why included[/bold]")
    if not bundle.items:
        console.print("  [dim]nothing ranked into the budget[/dim]")
    for i, item in enumerate(bundle.items, start=1):
        console.print(
            f"  [cyan]{i}.[/cyan] [bold]{item.id}[/bold]  "
            f"score={item.score:.4f}"
        )
        console.print(f"     reason: {item.reason or '—'}")
        meta_bits = []
        for key in (
            "rank_relevance",
            "rank_recency",
            "rank_graph",
            "rank_graph_distance",
            "compressed",
        ):
            if key in item.metadata:
                meta_bits.append(f"{key}={item.metadata[key]}")
        if meta_bits:
            console.print(f"     signals: {', '.join(meta_bits)}")

    gm = bundle.graph_metrics
    bm = bundle.bm25_metrics
    console.print("\n[bold]Metrics[/bold]")
    console.print(
        f"  bm25 top={bm.top_score:.4f} zero={bm.scores_zero} "
        f"coverage={bm.query_term_coverage:.2f} low_top={bm.low_top_score}"
    )
    console.print(
        f"  graph neighbors={gm.neighbor_calls} returned={gm.neighbor_nodes_returned} "
        f"empty={gm.neighbor_empty_results} slow={gm.slow_queries} "
        f"latency_avg_ms={gm.neighbor_latency_ms_avg:.2f}"
    )
    if bundle.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in bundle.notes:
            console.print(f"  [yellow]•[/yellow] {note}")


@context_app.command("retrieve")
def context_retrieve(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    query: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Free-text query for BM25 retrieve / rank",
    ),
    goal: str = typer.Option(
        "",
        "--goal",
        "-g",
        help="Session/task goal (defaults to query)",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="text (summary) or json (full ContextBundle)",
    ),
    max_items: int = typer.Option(16, "--max-items", help="Compress item budget"),
    max_chars: int = typer.Option(4000, "--max-chars", help="Compress char budget"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Build a ContextBundle from workspace / RI / experiment sources."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")
    bundle = _build_bundle(
        competition=competition,
        query=query,
        goal=goal,
        max_items=max_items,
        max_chars=max_chars,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    if output_format == "json":
        print(bundle.to_json(indent=2))
        return
    _print_summary(bundle)


@context_app.command("explain")
def context_explain(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    query: str = typer.Option(
        "",
        "--query",
        "-q",
        help="Free-text query for BM25 retrieve / rank",
    ),
    goal: str = typer.Option(
        "",
        "--goal",
        "-g",
        help="Session/task goal (defaults to query)",
    ),
    max_items: int = typer.Option(16, "--max-items", help="Compress item budget"),
    max_chars: int = typer.Option(4000, "--max-chars", help="Compress char budget"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Show ranked evidence with inclusion reasons and rank signals."""
    bundle = _build_bundle(
        competition=competition,
        query=query,
        goal=goal,
        max_items=max_items,
        max_chars=max_chars,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    _print_explain(bundle)
