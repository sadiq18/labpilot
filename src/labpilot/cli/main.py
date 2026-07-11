import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from labpilot.config import LLMConfig, load_config
from labpilot.diagnostics import check_environment, print_diagnostics_report
from labpilot.llm.client import LLMClient, create_llm_client
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
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Run the full research pipeline for a Kaggle competition.

    For a two-step workflow that pauses after the brief to review the
    resolved competition/baseline choice before training, use `research
    init` followed by `research build` instead.
    """
    _fail_fast_on_bad_environment()

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(f"[bold]LabPilot[/bold] — starting run for [cyan]{competition}[/cyan]\n")

    pipeline = Pipeline(config, submit=submit, configs_dir=competitions_dir, llm_client=llm_client)
    manifest = pipeline.run(competition)

    console.print(f"\n[green]Run complete:[/green] {config.runs_dir / manifest.run_id}")
    console.print(
        f"[green]Reflection:[/green] {config.runs_dir / manifest.run_id / 'reflection.md'}"
    )


@app.command()
def init(
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
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Run just the init half: parse competition → download data → profile → brief.

    Stops before any baseline is chosen or trained, so you can inspect
    competition.json/profile.json/brief.md and then run `research build
    --run-id <id>` to continue.
    """
    _fail_fast_on_bad_environment(skip_lightgbm=True)

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(f"[bold]LabPilot[/bold] — initializing run for [cyan]{competition}[/cyan]\n")

    pipeline = Pipeline(config, configs_dir=competitions_dir, llm_client=llm_client)
    manifest = pipeline.init(competition)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Init complete:[/green] {run_dir}")
    console.print(f"[green]Brief:[/green] {run_dir / 'brief.md'}")
    console.print(f"\nNext: [cyan]research build --run-id {manifest.run_id}[/cyan]")


@app.command()
def build(
    run_id: str = typer.Option(
        ..., "--run-id", "-r", help="Run ID to build (from `research init`)"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(
        None, "--runs-dir", help="Override runs directory (must match the `init` call's)"
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Upload the validated submission to Kaggle (disabled by default)",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Run the build half of an already-`init`'d run: baseline through reflection."""
    _fail_fast_on_bad_environment()

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)
    console.print(f"[bold]LabPilot[/bold] — building run [cyan]{run_id}[/cyan]\n")

    pipeline = Pipeline(config, submit=submit, llm_client=llm_client)
    manifest = _continue_or_exit(pipeline.build, run_id)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Build complete:[/green] {run_dir}")
    console.print(f"[green]Reflection:[/green] {run_dir / 'reflection.md'}")


@app.command()
def resume(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to resume"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(
        None, "--runs-dir", help="Override runs directory (must match the original run's)"
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
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Resume a run from its first failed or incomplete stage.

    Stages already marked completed or skipped are left untouched; everything
    else (failed, still "running" from a killed process, or never reached) is
    re-executed in pipeline order.
    """
    _fail_fast_on_bad_environment()

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)
    console.print(f"[bold]LabPilot[/bold] — resuming run [cyan]{run_id}[/cyan]\n")

    pipeline = Pipeline(config, submit=submit, configs_dir=competitions_dir, llm_client=llm_client)
    manifest = _continue_or_exit(pipeline.resume, run_id)

    console.print(f"\n[green]Run complete:[/green] {config.runs_dir / manifest.run_id}")


def _continue_or_exit(action, run_id: str):
    """Shared error handling for `build`/`resume`: turn a missing run or an
    unmet precondition (e.g. `build` before `init` finished) into a clean
    one-line error instead of a raw traceback.
    """
    try:
        return action(run_id)
    except FileNotFoundError:
        console.print(f"[red]Run not found:[/red] {run_id} (check --runs-dir).")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None


def _check_llm_or_confirm(config: LLMConfig, assume_yes: bool) -> LLMClient | None:
    """Check LLM availability once up front instead of silently falling
    back mid-run: warn when no provider is available, and let an
    interactive user opt out before spending 1-4 hours on a run that will
    only produce template-text briefs/reflections.

    Non-interactive runs (CI, scripts, `--yes`) never block on a prompt —
    they just warn and proceed, preserving the "unattended for hours"
    promise. A missing LLM is never treated as a hard failure the way
    `_fail_fast_on_bad_environment` treats a broken Python/LightGBM/Kaggle
    setup, since an LLM is optional by design.
    """
    client = create_llm_client(config)
    if client is not None:
        return client

    console.print(
        "[yellow]No LLM provider available[/yellow] (missing OPENAI_API_KEY/GEMINI_API_KEY, "
        "or the optional package isn't installed) — brief.md/reflection.md will use "
        "template fallback text instead of AI-generated content."
    )
    if assume_yes or not sys.stdin.isatty():
        console.print("Proceeding without LLM.")
        return None

    if Confirm.ask("Continue without LLM-generated brief/reflection?", default=True):
        return None

    console.print("[red]Aborted.[/red]")
    raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Check that the local environment has everything LabPilot needs."""
    results = check_environment()
    all_ok = print_diagnostics_report(results, console)
    if not all_ok:
        raise typer.Exit(code=1)


def _fail_fast_on_bad_environment(skip_lightgbm: bool = False) -> None:
    # `research init` never touches LightGBM (no baseline is trained), so a
    # broken/missing install on this machine shouldn't block it — only
    # `build`/`run`/`resume`, which actually train a model, need to check.
    results = [
        r for r in check_environment() if not (skip_lightgbm and r.name == "LightGBM import")
    ]
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Show the status of a research run."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
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
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """List all research runs."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    runs_root = config.runs_dir

    if not runs_root.exists():
        console.print("No runs found.")
        raise typer.Exit()

    table = Table(title="Research Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Competition")
    table.add_column("Status")

    status_styles = {
        StageStatus.COMPLETED: "green",
        StageStatus.FAILED: "red",
        StageStatus.RUNNING: "yellow",
        StageStatus.PARTIAL: "cyan",
    }
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = find_manifest(config, run_dir.name)
            style = status_styles.get(manifest.status, "dim")
            table.add_row(
                manifest.run_id, manifest.competition, f"[{style}]{manifest.status.value}[/{style}]"
            )

    console.print(table)


if __name__ == "__main__":
    app()
