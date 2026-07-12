import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from labpilot.competition.models import CompetitionSpec
from labpilot.config import LLMConfig, load_config
from labpilot.diagnostics import (
    check_environment,
    print_diagnostics_report,
    required_environment_checks,
)
from labpilot.kaggle.client import SubmissionResult
from labpilot.llm.client import LLMClient, create_llm_client
from labpilot.orchestrator.manifest import StageStatus
from labpilot.orchestrator.pipeline import Pipeline, find_manifest
from labpilot.tracking.index import diff_runs

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = typer.Typer(
    name="research",
    help="LabPilot — one-command Kaggle competition research engine",
    no_args_is_help=True,
)
runs_app = typer.Typer(help="Inspect and compare research runs.")
app.add_typer(runs_app, name="runs")
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


def _validate_submit_flags(submit: bool, force_submit: bool) -> None:
    if force_submit and not submit:
        raise typer.BadParameter("--force-submit requires --submit.")


def _print_kernel_init_banner(run_dir: Path) -> None:
    competition_path = run_dir / "competition.json"
    if not competition_path.is_file():
        return
    competition = CompetitionSpec.model_validate_json(competition_path.read_text())
    if competition.submission_mode != "kernel":
        return
    console.print(
        "\n[yellow]Kernel-only competition:[/yellow] LabPilot will train locally and "
        "submit via the Kaggle notebook API when you pass --submit."
    )
    if competition.submissions_url:
        console.print(f"[bold]Submissions:[/bold] {competition.submissions_url}")
    console.print(
        "Pass [cyan]--submit[/cyan] on `research build` or `research resume` to push the kernel."
    )


def _print_submission_links_if_present(run_dir: Path, submit: bool) -> None:
    if not submit:
        return
    result_path = run_dir / "submission_result.json"
    if not result_path.is_file():
        return
    result = SubmissionResult.model_validate_json(result_path.read_text())
    if result.submissions_url:
        console.print(f"\n[bold]Submissions:[/bold] {result.submissions_url}")
    if result.kernel_url:
        console.print(f"[bold]Kernel:[/bold]      {result.kernel_url}")


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
    force_submit: bool = typer.Option(
        False,
        "--force-submit",
        help="With --submit: upload even when the competition deadline has passed",
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
    _validate_submit_flags(submit, force_submit)

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(f"[bold]LabPilot[/bold] — starting run for [cyan]{competition}[/cyan]\n")

    pipeline = Pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        configs_dir=competitions_dir,
        llm_client=llm_client,
    )
    manifest = pipeline.run(competition)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Run complete:[/green] {run_dir}")
    console.print(
        f"[green]Reflection:[/green] {run_dir / 'reflection.md'}"
    )
    _print_submission_links_if_present(run_dir, submit)


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
    _print_kernel_init_banner(run_dir)
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
    force_submit: bool = typer.Option(
        False,
        "--force-submit",
        help="With --submit: upload even when the competition deadline has passed",
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
    _validate_submit_flags(submit, force_submit)

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)
    console.print(f"[bold]LabPilot[/bold] — building run [cyan]{run_id}[/cyan]\n")

    pipeline = Pipeline(
        config, submit=submit, force_submit=force_submit, llm_client=llm_client
    )
    manifest = _continue_or_exit(pipeline.build, run_id)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Build complete:[/green] {run_dir}")
    console.print(f"[green]Reflection:[/green] {run_dir / 'reflection.md'}")
    _print_submission_links_if_present(run_dir, submit)


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
    force_submit: bool = typer.Option(
        False,
        "--force-submit",
        help="With --submit: upload even when the competition deadline has passed",
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
    _validate_submit_flags(submit, force_submit)

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)
    console.print(f"[bold]LabPilot[/bold] — resuming run [cyan]{run_id}[/cyan]\n")

    pipeline = Pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        configs_dir=competitions_dir,
        llm_client=llm_client,
    )
    manifest = _continue_or_exit(pipeline.resume, run_id)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Run complete:[/green] {run_dir}")
    _print_submission_links_if_present(run_dir, submit)


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
    # Optional extras (image/deep torch stacks) are reported by `research doctor`
    # but must not block tabular/text lightweight runs.
    results = required_environment_checks(skip_lightgbm=skip_lightgbm)
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
def improve(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Parent run ID to improve"),
    strategy: str = typer.Option(
        "auto",
        "--strategy",
        help="Improvement strategy: auto (LLM plan), tune (LightGBM grid), or features",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Upload the validated submission to Kaggle (disabled by default)",
    ),
    force_submit: bool = typer.Option(
        False,
        "--force-submit",
        help="With --submit: upload even when the competition deadline has passed",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Fork a completed run, apply an improvement plan, and retrain."""
    _fail_fast_on_bad_environment()
    _validate_submit_flags(submit, force_submit)

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(
        f"[bold]LabPilot[/bold] — improving run [cyan]{run_id}[/cyan] "
        f"(strategy={strategy})\n"
    )

    pipeline = Pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        llm_client=llm_client,
    )
    manifest = _continue_or_exit(lambda _: pipeline.improve(run_id, strategy=strategy), run_id)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Improvement complete:[/green] {run_dir}")
    console.print(f"[green]Reflection:[/green] {run_dir / 'reflection.md'}")
    console.print(
        f"\nCompare: [cyan]research runs diff --base {run_id} --compare {manifest.run_id}[/cyan]"
    )
    _print_submission_links_if_present(run_dir, submit)


@runs_app.command("diff")
def runs_diff(
    base: str = typer.Option(..., "--base", help="Base (parent) run ID"),
    compare: str = typer.Option(..., "--compare", help="Run ID to compare against base"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Compare two runs side-by-side (metrics, params, lineage)."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir

    try:
        result = diff_runs(config.runs_dir, base, compare)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(f"\n[bold]Run diff[/bold]: [cyan]{base}[/cyan] → [cyan]{compare}[/cyan]\n")

    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Base")
    metrics_table.add_column("Compare")
    metrics_table.add_column("Delta")
    all_metrics = sorted(set(result.base_metrics) | set(result.compare_metrics))
    for key in all_metrics:
        base_val = result.base_metrics.get(key)
        compare_val = result.compare_metrics.get(key)
        delta = result.metric_deltas.get(key)
        delta_str = f"{delta:+.6f}" if delta is not None else "-"
        metrics_table.add_row(
            key,
            f"{base_val:.6f}" if isinstance(base_val, (int, float)) else "-",
            f"{compare_val:.6f}" if isinstance(compare_val, (int, float)) else "-",
            delta_str,
        )
    console.print(metrics_table)

    if result.param_changes:
        params_table = Table(title="Param changes")
        params_table.add_column("Param", style="cyan")
        params_table.add_column("Base")
        params_table.add_column("Compare")
        for key, change in result.param_changes.items():
            params_table.add_row(key, str(change.get("base")), str(change.get("compare")))
        console.print(params_table)

    lineage_table = Table(title="Lineage")
    lineage_table.add_column("Field", style="cyan")
    lineage_table.add_column("Value")
    for key, value in result.lineage.items():
        lineage_table.add_row(key, str(value))
    console.print(lineage_table)

    submission_table = Table(title="Submission status")
    submission_table.add_column("Run", style="cyan")
    submission_table.add_column("Status")
    submission_table.add_row(base, result.submission_notes.get("base", "-"))
    submission_table.add_row(compare, result.submission_notes.get("compare", "-"))
    console.print(submission_table)


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
