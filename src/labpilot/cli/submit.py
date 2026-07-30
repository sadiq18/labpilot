"""``research submit`` — upload execution-scoped CSV and record LB learning."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.research_engine.execution.submit_learn import (
    SubmitLearnError,
    submit_and_learn,
)

console = Console()


def submit_command(
    *,
    execution_id: str,
    competition: str | None,
    config_path: Path,
    knowledge_dir: Path | None,
    workspace_path: Path | None,
    path: Path | None,
    message: str | None,
    dry_run: bool,
) -> None:
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    root = None
    if workspace is not None:
        root = workspace.root

    try:
        summary = submit_and_learn(
            knowledge_dir=config.knowledge_dir,
            competition=competition,
            execution_id=execution_id,
            workspace_root=root,
            submission_path=path,
            message=message,
            kaggle_config=config.kaggle,
            dry_run=dry_run,
        )
    except SubmitLearnError as exc:
        console.print(f"[red]Submit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    lb = summary.leaderboard
    console.print(
        f"[bold]Submit[/bold] — execution [cyan]{execution_id}[/cyan] "
        f"({'dry-run' if dry_run else 'uploaded'})"
    )
    if summary.submission_path:
        console.print(f"  file: {summary.submission_path}")
    if lb and lb.public_score is not None:
        console.print(f"  public_score: [green]{lb.public_score}[/green]")
        if lb.prior_public_score is not None:
            console.print(f"  prior_public: {lb.prior_public_score}")
        if lb.delta_vs_prior is not None:
            console.print(f"  delta_vs_prior: {lb.delta_vs_prior:+.6g}")
        if lb.overfitting:
            console.print(
                "  [yellow]overfit signal[/yellow]: local CV strong, public LB weak"
            )
    elif not dry_run:
        console.print("  public_score: (pending / unscored)")
    if summary.follow_up_hypothesis_id:
        console.print(
            f"  improvement hypothesis: [cyan]{summary.follow_up_hypothesis_id}[/cyan]"
        )
    if lb and lb.submissions_url:
        console.print(f"\n[bold]Submissions:[/bold] {lb.submissions_url}")
