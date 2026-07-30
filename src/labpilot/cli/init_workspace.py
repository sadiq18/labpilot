"""``research init`` — scaffold a client-owned competition workspace."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

from labpilot.research_engine.intelligence.context import normalize_competition
from labpilot.workspace import init_git_repo, scaffold_workspace

console = Console()


def init_workspace_command(
    *,
    competition: str,
    path: Path,
    git: bool | None,
    force: bool,
    labpilot_hint: Path | None,
) -> None:
    """Create ``<path>/<slug>/`` with labpilot.yaml, dirs, gitignore, optional git."""
    try:
        slug, _url = normalize_competition(competition)
    except ValueError as exc:
        console.print(f"[red]Invalid competition:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    root = (Path(path).expanduser().resolve() / slug)
    hint = str((labpilot_hint or Path.cwd()).resolve())

    try:
        workspace = scaffold_workspace(
            root,
            slug,
            force=force,
            labpilot_hint=hint,
        )
    except FileExistsError as exc:
        console.print(f"[red]Refused:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        console.print(f"[red]Could not create workspace:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    do_git = git
    if do_git is None:
        do_git = Confirm.ask(
            f"Initialize git repo in [cyan]{workspace.root}[/cyan]?",
            default=False,
        )

    if do_git:
        try:
            init_git_repo(workspace.root)
            console.print("[green]git[/green] initialized with scaffold commit")
        except Exception as exc:
            console.print(f"[yellow]git init skipped:[/yellow] {exc}")

    console.print(
        f"[bold green]Created workspace[/bold green] [cyan]{workspace.root}[/cyan]\n"
        f"  competition: {workspace.competition}\n"
        f"  marker: {workspace.marker_path.name}"
    )
    console.print(
        "\n[bold]Next[/bold] (artifacts stay in this folder):\n"
        f"  cd {workspace.root}\n"
        f"  cp .env.example .env   # then set KAGGLE_API_TOKEN\n"
        f"  uv run --project {hint} research analyze\n"
        f"  uv run --project {hint} research plan create --baseline\n"
        f"  uv run --project {hint} research run --plan P-001\n"
    )
