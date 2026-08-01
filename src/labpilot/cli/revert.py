"""CLI: restore workspace code to an experiment's recorded git commit."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import load_cli_config, resolve_competition, resolve_os_workspace
from labpilot.research_engine.agents.git_evolution import (
    find_experiment_record,
    revert_to_commit,
)

console = Console()


def revert_command(
    experiment_id: str,
    *,
    config_path: Path | None = None,
    knowledge_dir: Path | None = None,
    runs_dir: Path | None = None,
    workspace_path: Path | None = None,
    competition: str | None = None,
) -> None:
    """Restore code paths to the git commit stored on an experiment record."""
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
        workspace_path=workspace_path,
    )
    slug = resolve_competition(competition, client, required=True)
    ws = resolve_os_workspace(competition=slug, config=config, client=client)

    record = find_experiment_record(ws.root, experiment_id)
    if record is None:
        console.print(f"[red]No experiment record for[/red] {experiment_id!r}")
        raise typer.Exit(code=1)

    commit = record.get("git_commit")
    if not commit:
        console.print(
            f"[red]Experiment[/red] {experiment_id!r} [red]has no git_commit[/red]"
        )
        raise typer.Exit(code=1)

    try:
        revert_to_commit(ws.root, str(commit))
    except RuntimeError as exc:
        console.print(f"[red]Revert failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    short = str(commit)[:7]
    console.print(
        f"[green]Restored code[/green] to {short} "
        f"(execution={record.get('execution_id')}, branch={record.get('git_branch')})"
    )
