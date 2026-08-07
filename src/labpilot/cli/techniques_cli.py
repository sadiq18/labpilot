"""CLI for technique vocabulary status (M-25 step 1)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.research_engine.execution.technique.vocabulary import (
    format_technique_status_report,
    load_aging_context,
    recompute_technique_status,
    technique_status_report,
)

techniques_app = typer.Typer(
    help="Technique vocabulary — derived status from evidence cards.",
    no_args_is_help=True,
)
console = Console()


@techniques_app.command("report")
def techniques_report(
    competition: str | None = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    output_json: bool = typer.Option(False, "--json"),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Write derived statuses (default: report only)",
    ),
) -> None:
    """Show what status recompute would do — step 1 review before filtering."""
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    slug = resolve_competition(competition, workspace)
    aging = load_aging_context(config.knowledge_dir, slug)
    if apply:
        changed = recompute_technique_status(
            config.knowledge_dir, slug, aging=aging
        )
        console.print(
            f"[green]Updated status for {len(changed)} technique(s)[/green] "
            f"in [cyan]{slug}[/cyan]"
        )
    report = technique_status_report(config.knowledge_dir, slug, aging=aging)
    if output_json:
        print(json.dumps(report, indent=2))
        return
    console.print(format_technique_status_report(report))
