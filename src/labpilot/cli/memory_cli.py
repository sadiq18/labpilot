"""``research memory`` — seed / inspect / browse shared experience memory.

Memory influences Conductor via ContextBundle (retrieve-always). ``seed`` is an
explicit operator action — never run automatically from campaign start.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.research_engine.context import ContextRequest, ExperienceProvider, build_context
from labpilot.research_engine.memory import ExperienceStore
from labpilot.research_engine.memory.seed import write_seed_manifest

memory_app = typer.Typer(
    help=(
        "Shared experience memory (cross-competition). "
        "Influences decisions via ContextBundle; seed is operator-driven."
    ),
    no_args_is_help=True,
)
console = Console()


def _store_for(
    *,
    config_path: Path,
    knowledge_dir: Path | None,
    runs_dir: Path | None,
) -> tuple[ExperienceStore, Path]:
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    return ExperienceStore(config.knowledge_dir, workspace=workspace), config.knowledge_dir


@memory_app.command("seed")
def memory_seed(
    source: str = typer.Option(
        ...,
        "--from",
        help="Source competition slug whose experiences to attach as priors",
    ),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Target competition (workspace default)",
    ),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Explicitly seed priors from another competition into this workspace.

    Writes an auditable manifest under the target competition. Does not change
    Conductor policy by itself — Context Engine may boost seeded experience ids.
    """
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    target = resolve_competition(competition, workspace)
    source_slug = source.strip()
    if not source_slug:
        raise typer.BadParameter("--from requires a competition slug")
    if source_slug == target:
        console.print(
            "[yellow]Source and target are the same slug — seed is a no-op copy.[/yellow]"
        )

    store = ExperienceStore(config.knowledge_dir, workspace=workspace)
    try:
        records = store.list(source_competition=source_slug)
        if not records:
            console.print(
                f"[yellow]No experiences found for[/yellow] [cyan]{source_slug}[/cyan]"
            )
            raise typer.Exit(code=1)
        path = write_seed_manifest(
            config.knowledge_dir,
            target_competition=target,
            source_competition=source_slug,
            records=records,
        )
    finally:
        store.close()

    console.print(
        f"[bold green]Seeded[/bold green] {len(records)} experience(s) from "
        f"[cyan]{source_slug}[/cyan] → [cyan]{target}[/cyan]\n"
        f"  manifest: {path}\n"
        f"[dim]Memory influences via ContextBundle; seed is operator-driven.[/dim]"
    )


@memory_app.command("inspect")
def memory_inspect(
    similar_to: str = typer.Option(
        ...,
        "--similar-to",
        help="Competition slug (or query focus) to find similar experiences for",
    ),
    query: str = typer.Option("", "--query", "-q", help="Optional lexical query"),
    limit: int = typer.Option(16, "--limit", "-n", help="Max context items to show"),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Request competition for Context Engine (workspace default)",
    ),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Show what retrieve would surface from experience memory (trust/debug)."""
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    slug = resolve_competition(competition, workspace)
    focus = similar_to.strip()
    q = (query or focus).strip()
    request = ContextRequest(
        competition=slug,
        goal=f"Transfer priors similar to {focus}",
        query=q,
        knowledge_dir=config.knowledge_dir,
        max_items=limit,
        max_chars=8000,
        metadata={"experience_limit": 200},
    )
    bundle = build_context(request, providers=[ExperienceProvider()])
    items = [i for i in bundle.items if i.source == "experience"]

    console.print(
        f"\n[bold]Memory inspect[/bold] — similar to [cyan]{focus}[/cyan] "
        f"(request competition [cyan]{slug}[/cyan])"
    )
    console.print(
        "[dim]Shows ContextBundle experience items only — does not change Conductor strategy.[/dim]"
    )
    if bundle.provider_errors:
        for err in bundle.provider_errors:
            console.print(f"[red]provider:[/red] {err}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="cyan")
    table.add_column("outcome", width=8)
    table.add_column("from", width=16)
    table.add_column("facets")
    table.add_column("score", justify="right", width=7)
    table.add_column("artifacts")
    for item in items:
        meta = item.metadata or {}
        exp_id = str(meta.get("experience_id") or item.id.replace("experience:", ""))
        facets = ", ".join(str(f) for f in (meta.get("facets") or [])[:6])
        arts = meta.get("artifacts") or {}
        art_bits = []
        if arts.get("experiment_id"):
            art_bits.append(f"exp={arts['experiment_id']}")
        if arts.get("git_commit"):
            art_bits.append(f"git={str(arts['git_commit'])[:8]}")
        if arts.get("reflection_id"):
            art_bits.append(f"refl={arts['reflection_id']}")
        table.add_row(
            exp_id,
            str(meta.get("outcome") or meta.get("status") or "—"),
            str(meta.get("source_competition") or "—"),
            facets or "—",
            f"{item.score:.3f}",
            ", ".join(art_bits) or "—",
        )
    if items:
        console.print(table)
    else:
        console.print("[dim](no similar experiences in ContextBundle)[/dim]")


@memory_app.command("list")
def memory_list(
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Filter by source competition slug",
    ),
    outcome: str | None = typer.Option(
        None,
        "--outcome",
        help="Filter outcome: success|fail",
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Filter by facet name (alias of facet)",
    ),
    limit: int | None = typer.Option(50, "--limit", "-n"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """List experiences in the shared ExperienceStore."""
    if outcome is not None and outcome not in {"success", "fail"}:
        raise typer.BadParameter("--outcome must be success or fail")
    store, _ = _store_for(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    try:
        db_path = store.db_path
        records = store.list(
            source_competition=competition,
            outcome=outcome,  # type: ignore[arg-type]
            tag=tag,
            limit=limit,
        )
    finally:
        store.close()

    table = Table(show_header=True, header_style="bold")
    table.add_column("id", style="cyan")
    table.add_column("source")
    table.add_column("outcome", width=8)
    table.add_column("facets")
    table.add_column("goal")
    for record in records:
        table.add_row(
            record.id,
            record.source_competition,
            record.outcome,
            ", ".join(record.facet_names()[:6]) or "—",
            (record.goal or "")[:60] or "—",
        )
    console.print(table if records else "[dim](no experiences)[/dim]")
    console.print(f"[dim]{len(records)} row(s) — db={db_path}[/dim]")


@memory_app.command("show")
def memory_show(
    experience_id: str = typer.Argument(..., help="Experience id (XR-001)"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    runs_dir: Path | None = typer.Option(None, "--runs-dir"),
) -> None:
    """Show one Experience Record."""
    store, _ = _store_for(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    try:
        record = store.get(experience_id.strip())
    finally:
        store.close()
    if record is None:
        console.print(f"[red]Unknown experience[/red] {experience_id}")
        raise typer.Exit(code=1)

    console.print(f"[bold]{record.id}[/bold]  ({record.outcome})  from [cyan]{record.source_competition}[/cyan]")
    console.print(f"  goal:        {record.goal or '—'}")
    console.print(f"  hypothesis:  {record.hypothesis or '—'}")
    console.print(f"  action:      {record.action or '—'}")
    console.print(f"  result:      {record.result or '—'}")
    console.print(f"  idempotency: {record.idempotency_key}")
    console.print("  facets:")
    if not record.facets:
        console.print("    —")
    for facet in record.facets:
        console.print(
            f"    • {facet.facet}  conf={facet.confidence:.2f}  "
            f"source={facet.source}  evidence={facet.evidence}"
        )
    arts = record.artifacts
    console.print(
        "  artifacts: "
        f"experiment={arts.experiment_id or '—'}  "
        f"execution={arts.execution_id or '—'}  "
        f"git={arts.git_commit or '—'}  "
        f"reflection={arts.reflection_id or '—'}"
    )
