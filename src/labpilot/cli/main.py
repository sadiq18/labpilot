from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.config import load_config
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline, find_manifest

app = typer.Typer(
    name="research",
    help="LabPilot — one-command Kaggle competition research engine",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Run the full research pipeline for a Kaggle competition."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir

    console.print(f"[bold]LabPilot[/bold] — starting run for [cyan]{competition}[/cyan]\n")

    pipeline = Pipeline(config)
    manifest = pipeline.run(competition)

    console.print(f"\n[green]Run complete:[/green] {config.runs_dir / manifest.run_id}")
    console.print(f"[green]Reflection:[/green] {config.runs_dir / manifest.run_id / 'reflection.md'}")


@app.command()
def status(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to inspect"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
) -> None:
    """Show the status of a research run."""
    config = load_config(config_path)
    manifest = find_manifest(config, run_id)

    table = Table(title=f"Run: {manifest.run_id}")
    table.add_column("Stage", style="cyan")
    table.add_column("Status")
    table.add_column("Artifacts")

    for stage in manifest.stages:
        status_style = {
            StageStatus.COMPLETED: "green",
            StageStatus.FAILED: "red",
            StageStatus.RUNNING: "yellow",
        }.get(stage.status, "dim")
        table.add_row(
            stage.name,
            f"[{status_style}]{stage.status.value}[/{status_style}]",
            ", ".join(stage.artifacts) if stage.artifacts else "-",
        )

    console.print(table)


@app.command()
def list_runs(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
) -> None:
    """List all research runs."""
    config = load_config(config_path)
    runs_root = config.runs_dir

    if not runs_root.exists():
        console.print("No runs found.")
        raise typer.Exit()

    table = Table(title="Research Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Competition")
    table.add_column("Status")

    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = find_manifest(config, run_dir.name)
            table.add_row(manifest.run_id, manifest.competition, manifest.status.value)

    console.print(table)


if __name__ == "__main__":
    app()
