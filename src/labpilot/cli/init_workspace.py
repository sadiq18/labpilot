"""``research init`` — scaffold a client-owned competition workspace."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from labpilot.research_engine.intelligence.context import normalize_competition
from labpilot.workspace import (
    EXPERIENCE_DB_ENV,
    EXPERIENCE_DB_FILENAME,
    MARKER_NAME,
    USER_EXPERIENCE_DB,
    CompetitionWorkspace,
    init_git_repo,
    resolve_experience_db_path,
    scaffold_workspace,
    update_workspace_experience_path,
)

console = Console()


def _normalize_experience_path(raw: Path | str) -> Path:
    path = Path(raw).expanduser().resolve()
    if path.is_dir() or not path.suffix:
        path = path / EXPERIENCE_DB_FILENAME
    return path


def _print_experience_export_hint(path: Path) -> None:
    console.print(
        f"\n[dim]Optional — pin this machine-wide:[/dim]\n"
        f"  export {EXPERIENCE_DB_ENV}={path}"
    )


def prompt_experience_db_fallback(default_fallback: Path) -> Path:
    """Ask before using ``~/.labpilot/experiences.db`` (or accept a custom path)."""
    console.print(
        "\n[bold yellow]Experience memory — fallback required[/bold yellow]\n"
        "No shared research-root / env / yaml path is available.\n"
        f"Fallback location: [cyan]{default_fallback}[/cyan]\n"
    )
    choice = Prompt.ask(
        f"Use fallback, or provide a custom path (set {EXPERIENCE_DB_ENV})?",
        choices=["fallback", "custom"],
        default="custom",
    )
    if choice == "custom":
        raw = Prompt.ask(
            "Path for experiences.db",
            default=str(default_fallback),
        ).strip()
        if not raw:
            raise typer.BadParameter("Experience DB path is required")
        path = _normalize_experience_path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]Using custom experience DB[/green] [cyan]{path}[/cyan]")
        _print_experience_export_hint(path)
        return path

    if not Confirm.ask(
        f"Confirm fallback to [cyan]{default_fallback}[/cyan]?",
        default=False,
    ):
        console.print(
            "[red]Aborted.[/red] Re-run and choose [bold]custom[/bold], or set "
            f"[cyan]{EXPERIENCE_DB_ENV}[/cyan] before setup."
        )
        raise typer.Exit(code=1)

    default_fallback.parent.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[yellow]Using fallback experience DB[/yellow] [cyan]{default_fallback}[/cyan]"
    )
    _print_experience_export_hint(default_fallback)
    return default_fallback.resolve()


def configure_experience_db_for_init(
    workspace: CompetitionWorkspace,
    *,
    experience_db: Path | None,
    experience_db_fallback: bool,
    interactive: bool,
) -> CompetitionWorkspace:
    """Bind shared experiences.db during workspace setup.

    Falling back to ``~/.labpilot`` requires explicit agreement (interactive
    confirm) or ``--experience-db-fallback``.
    """
    if experience_db is not None:
        path = _normalize_experience_path(experience_db)
        path.parent.mkdir(parents=True, exist_ok=True)
        updated = update_workspace_experience_path(workspace, path)
        console.print(
            f"[green]Experience DB[/green] [cyan]{path}[/cyan] "
            f"(written to {MARKER_NAME})"
        )
        _print_experience_export_hint(path)
        return updated

    env_raw = os.environ.get(EXPERIENCE_DB_ENV, "").strip()
    if env_raw:
        path = _normalize_experience_path(env_raw)
        updated = update_workspace_experience_path(workspace, path)
        console.print(
            f"[green]Experience DB[/green] from [cyan]{EXPERIENCE_DB_ENV}[/cyan]: "
            f"[cyan]{path}[/cyan]"
        )
        return updated

    parent_default = (workspace.research_root_parent / EXPERIENCE_DB_FILENAME).resolve()
    relative_parent = f"../{EXPERIENCE_DB_FILENAME}"

    if experience_db_fallback:
        path = USER_EXPERIENCE_DB.resolve()
        if interactive and not Confirm.ask(
            f"Use fallback experience DB [cyan]{path}[/cyan]?",
            default=False,
        ):
            console.print(
                "[red]Aborted.[/red] Pass [cyan]--experience-db PATH[/cyan] "
                f"or set [cyan]{EXPERIENCE_DB_ENV}[/cyan]."
            )
            raise typer.Exit(code=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        updated = update_workspace_experience_path(workspace, path)
        console.print(
            f"[yellow]Experience DB fallback[/yellow] [cyan]{path}[/cyan]"
        )
        _print_experience_export_hint(path)
        return updated

    if not interactive:
        # Keep scaffold parent-root default already written to yaml.
        console.print(
            f"[dim]Experience DB[/dim] [cyan]{parent_default}[/cyan] "
            "(parent research root; non-interactive default)"
        )
        return workspace

    console.print(
        "\n[bold]Shared experience memory[/bold] "
        "(cross-competition; not under this workspace)\n"
        f"  Suggested parent path: [cyan]{parent_default}[/cyan]\n"
        f"  User fallback:         [cyan]{USER_EXPERIENCE_DB}[/cyan]\n"
    )
    choice = Prompt.ask(
        "Where should experiences.db live?",
        choices=["parent", "custom", "fallback"],
        default="parent",
    )

    if choice == "parent":
        parent_default.parent.mkdir(parents=True, exist_ok=True)
        updated = update_workspace_experience_path(
            workspace, parent_default, store_as=relative_parent
        )
        console.print(
            f"[green]Experience DB[/green] [cyan]{parent_default}[/cyan] "
            "(parent research root)"
        )
        _print_experience_export_hint(parent_default)
        return updated

    if choice == "custom":
        raw = Prompt.ask("Path for experiences.db").strip()
        if not raw:
            console.print("[red]Path required.[/red]")
            raise typer.Exit(code=1)
        path = _normalize_experience_path(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        updated = update_workspace_experience_path(workspace, path)
        console.print(f"[green]Experience DB[/green] [cyan]{path}[/cyan]")
        _print_experience_export_hint(path)
        return updated

    # fallback — require explicit confirmation inside prompt helper
    path = prompt_experience_db_fallback(USER_EXPERIENCE_DB.resolve())
    return update_workspace_experience_path(workspace, path)


def init_workspace_command(
    *,
    competition: str,
    path: Path,
    git: bool | None,
    force: bool,
    labpilot_hint: Path | None,
    experience_db: Path | None = None,
    experience_db_fallback: bool = False,
) -> None:
    """Create ``<path>/<slug>/`` with labpilot.yaml, dirs, gitignore, optional git."""
    try:
        slug, _url = normalize_competition(competition)
    except ValueError as exc:
        console.print(f"[red]Invalid competition:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    root = Path(path).expanduser().resolve() / slug
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

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    workspace = configure_experience_db_for_init(
        workspace,
        experience_db=experience_db,
        experience_db_fallback=experience_db_fallback,
        interactive=interactive,
    )

    do_git = git
    if do_git is None:
        if interactive:
            do_git = Confirm.ask(
                f"Initialize git repo in [cyan]{workspace.root}[/cyan]?",
                default=False,
            )
        else:
            do_git = False

    if do_git:
        try:
            init_git_repo(workspace.root)
            console.print("[green]git[/green] initialized with scaffold commit")
        except Exception as exc:
            console.print(f"[yellow]git init skipped:[/yellow] {exc}")

    exp_path = resolve_experience_db_path(workspace=workspace)
    console.print(
        f"[bold green]Created workspace[/bold green] [cyan]{workspace.root}[/cyan]\n"
        f"  competition: {workspace.competition}\n"
        f"  marker: {workspace.marker_path.name}\n"
        f"  experience DB: {exp_path}"
    )
    console.print(
        "\n[bold]Next[/bold] (artifacts stay in this folder):\n"
        f"  cd {workspace.root}\n"
        f"  cp .env.example .env   # then set KAGGLE_API_TOKEN\n"
        f"  uv run --project {hint} research analyze\n"
        f"  uv run --project {hint} research plan create --baseline\n"
        f"  uv run --project {hint} research run --plan P-001\n"
    )
