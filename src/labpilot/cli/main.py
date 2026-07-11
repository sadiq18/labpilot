import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.config import load_config
from labpilot.diagnostics import check_environment, print_diagnostics_report
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline, find_manifest

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = typer.Typer(
    name="research",
    help="LabPilot — one-command Kaggle competition research engine",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging for every stage."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only log warnings and errors."),
) -> None:
    """LabPilot — one-command Kaggle competition research engine."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive.")
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.getLogger("labpilot").setLevel(level)


@app.command()
def run(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    competitions_dir: Path | None = typer.Option(
        None,
        "--competitions-dir",
        help=(
            "Directory containing local per-competition contracts "
            "(<slug>.yaml). Defaults to configs/competitions. See "
            "configs/competitions/README.md."
        ),
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Upload the validated submission to Kaggle (disabled by default)",
    ),
) -> None:
    """Run the full research pipeline for a Kaggle competition."""
    _fail_fast_on_bad_environment()

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir

    console.print(f"[bold]LabPilot[/bold] — starting run for [cyan]{competition}[/cyan]\n")

    pipeline = Pipeline(config, submit=submit, configs_dir=competitions_dir)
    manifest = pipeline.run(competition)

    console.print(f"\n[green]Run complete:[/green] {config.runs_dir / manifest.run_id}")
    console.print(
        f"[green]Reflection:[/green] {config.runs_dir / manifest.run_id / 'reflection.md'}"
    )


@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to resume"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    competitions_dir: Path | None = typer.Option(
        None,
        "--competitions-dir",
        help="Directory containing local per-competition contracts. See run --help.",
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Upload the validated submission to Kaggle (disabled by default)",
    ),
) -> None:
    """Resume a run from its first failed or incomplete stage.

    Stages already marked completed or skipped are left untouched; everything
    else (failed, still "running" from a killed process, or never reached) is
    re-executed in pipeline order.
    """
    _fail_fast_on_bad_environment()

    config = load_config(config_path)
    console.print(f"[bold]LabPilot[/bold] — resuming run [cyan]{run_id}[/cyan]\n")

    pipeline = Pipeline(config, submit=submit, configs_dir=competitions_dir)
    manifest = pipeline.resume(run_id)

    console.print(f"\n[green]Run complete:[/green] {config.runs_dir / manifest.run_id}")


@app.command()
def doctor() -> None:
    """Check that the local environment has everything LabPilot needs."""
    results = check_environment()
    all_ok = print_diagnostics_report(results, console)
    if not all_ok:
        raise typer.Exit(code=1)


def _fail_fast_on_bad_environment() -> None:
    results = check_environment()
    if all(result.ok for result in results):
        return
    console.print("[red]Environment check failed — run `research doctor` for details.[/red]")
    print_diagnostics_report(results, console)
    raise typer.Exit(code=1)


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
