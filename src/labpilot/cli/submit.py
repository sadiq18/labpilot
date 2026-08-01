"""``research submit`` — upload execution-scoped CSV and record LB learning.

Invokes the ``submit_learn`` tool (Strangler Phase A).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import (
    default_tools,
    load_cli_config,
    resolve_competition,
    resolve_os_workspace,
)
from labpilot.research_engine.execution.submit_learn import SubmitLearnError

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
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, client)
    ws = resolve_os_workspace(competition=competition, config=config, client=client)

    try:
        result = default_tools().invoke(
            "submit_learn",
            ws,
            execution_id=execution_id,
            submission_path=path,
            message=message,
            kaggle_config=config.kaggle,
            dry_run=dry_run,
        )
    except SubmitLearnError as exc:
        console.print(f"[red]Submit failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]Submit[/bold] — execution [cyan]{execution_id}[/cyan] "
        f"({'dry-run' if dry_run else 'uploaded'})"
    )
    if result.data.get("submission_path"):
        console.print(f"  file: {result.data['submission_path']}")
    public_score = result.data.get("public_score")
    if public_score is not None:
        console.print(f"  public_score: [green]{public_score}[/green]")
        if result.data.get("prior_public_score") is not None:
            console.print(f"  prior_public: {result.data['prior_public_score']}")
        if result.data.get("delta_vs_prior") is not None:
            console.print(f"  delta_vs_prior: {result.data['delta_vs_prior']:+.6g}")
        if result.data.get("overfitting"):
            console.print(
                "  [yellow]overfit signal[/yellow]: local CV strong, public LB weak"
            )
    elif not dry_run:
        console.print("  public_score: (pending / unscored)")
    if result.data.get("follow_up_hypothesis_id"):
        console.print(
            f"  improvement hypothesis: [cyan]{result.data['follow_up_hypothesis_id']}[/cyan]"
        )
    if result.data.get("submissions_url"):
        console.print(f"\n[bold]Submissions:[/bold] {result.data['submissions_url']}")
