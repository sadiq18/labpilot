import logging
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from labpilot.baseline.registry import list_templates
from labpilot.competition.models import CompetitionSpec
from labpilot.config import (
    AppConfig,
    load_config,
    resolve_competitions_dir,
    resolve_runtimes_dir,
)
from labpilot.diagnostics import (
    check_environment,
    print_diagnostics_report,
    required_environment_checks,
)
from labpilot.experiments.comparator import compare, load_comparison, render_markdown
from labpilot.experiments.graph import assemble_experiment, build_graph
from labpilot.experiments.hypothesis import HypothesisStore, linked_experiments
from labpilot.experiments.knowledge import KnowledgeBase
from labpilot.experiments.models import HypothesisCreatedBy, HypothesisStatus, Verdict
from labpilot.experiments.ranking import RankingWeights, rank_candidates
from labpilot.experiments.report import (
    NoExperimentsError,
    build_report,
    render_report_text,
    write_dashboard,
)
from labpilot.experiments.search import (
    SearchFilters,
    load_comparisons,
    parse_duration,
    parse_key_value,
    parse_metric_threshold,
    search,
)
from labpilot.kaggle.client import SubmissionResult
from labpilot.llm.client import LLMClient, llm_setup_hints, resolve_llm_client
from labpilot.orchestrator.manifest import StageStatus, load_manifest
from labpilot.orchestrator.pipeline import Pipeline, find_manifest
from labpilot.report.generator import ReportGenerator
from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.hypothesis import HypothesisAssistant
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import (
    UnknownAnalyzerError,
    build_default_registry,
)
from labpilot.research_engine.intelligence.renderers.json import to_json, write_report
from labpilot.research_engine.intelligence.renderers.terminal import render_terminal
from labpilot.research_engine.intelligence.retrieval import (
    ContextBuilder,
    QueryType,
)
from labpilot.runtimes.doctor import check_all_runtimes
from labpilot.runtimes.registry import get_runtime, list_runtimes
from labpilot.runtimes.templates import runtime_to_yaml_dict, scaffold_runtime
from labpilot.tracking.index import diff_runs
from labpilot.workspace.discover import init_project, load_project

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = typer.Typer(
    name="research",
    help="LabPilot — one-command Kaggle competition research engine",
    no_args_is_help=True,
)
runs_app = typer.Typer(help="Inspect and compare research runs.")
app.add_typer(runs_app, name="runs")
workspace_app = typer.Typer(help="Manage multi-competition project workspaces.")
app.add_typer(workspace_app, name="workspace")
runtime_app = typer.Typer(help="Register and validate training runtimes.")
app.add_typer(runtime_app, name="runtime")
experiments_app = typer.Typer(help="Explore the experiment graph.")
app.add_typer(experiments_app, name="experiments")
knowledge_app = typer.Typer(help="Explore the accumulated experiment knowledge base.")
experiments_app.add_typer(knowledge_app, name="knowledge")
hypothesis_app = typer.Typer(help="Manage structured hypotheses.")
app.add_typer(hypothesis_app, name="hypothesis")
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


def _validate_dry_run_flags(dry_run: bool, submit: bool) -> None:
    if dry_run and submit:
        raise typer.BadParameter("--dry-run and --submit are mutually exclusive.")


def _load_app_config(
    config_path: Path,
    runs_dir: Path | None,
    project_dir: Path | None,
    knowledge_dir: Path | None = None,
) -> AppConfig:

    config = load_config(config_path, project_dir=project_dir)
    if runs_dir:
        config.runs_dir = runs_dir
    if knowledge_dir:
        config.knowledge_dir = knowledge_dir
    return config


def _build_pipeline(
    config,
    *,
    submit: bool = False,
    force_submit: bool = False,
    competitions_dir: Path | None = None,
    llm_client: LLMClient | None = None,
    dry_run: bool = False,
    project_dir: Path | None = None,
) -> Pipeline:
    return Pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        configs_dir=resolve_competitions_dir(
            config,
            competitions_dir,
            project_dir=project_dir,
        ),
        llm_client=llm_client,
        dry_run=dry_run,
        project_dir=project_dir,
    )


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
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    competitions_dir: Path | None = typer.Option(
        None,
        "--competitions-dir",
        help=(
            "Directory containing local per-competition contracts "
            "(<slug>.yaml). Defaults to configs/competitions. See "
            "configs/competitions/README.md."
        ),
    ),
    hypothesis_id: str | None = typer.Option(
        None,
        "--hypothesis",
        help="Hypothesis ID to test with this root run (e.g. H-001)",
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate through code generation without training or submission",
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
    _fail_fast_on_bad_environment(skip_lightgbm=dry_run)
    _validate_submit_flags(submit, force_submit)
    _validate_dry_run_flags(dry_run, submit)

    config = _load_app_config(config_path, runs_dir, project_dir, knowledge_dir)
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(f"[bold]LabPilot[/bold] — starting run for [cyan]{competition}[/cyan]\n")
    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] stopping after code generation.\n")

    pipeline = _build_pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        competitions_dir=competitions_dir,
        llm_client=llm_client,
        dry_run=dry_run,
        project_dir=project_dir,
    )
    try:
        manifest = pipeline.run(competition, hypothesis_id=hypothesis_id)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

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
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="No effect on init-only workflow; reserved for symmetry with run/build",
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

    config = _load_app_config(config_path, runs_dir, project_dir)
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(f"[bold]LabPilot[/bold] — initializing run for [cyan]{competition}[/cyan]\n")

    pipeline = _build_pipeline(
        config,
        competitions_dir=competitions_dir,
        llm_client=llm_client,
        dry_run=dry_run,
        project_dir=project_dir,
    )
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
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate through code generation without training or submission",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Run the build half of an already-`init`'d run: baseline through reflection."""
    _fail_fast_on_bad_environment(skip_lightgbm=dry_run)
    _validate_submit_flags(submit, force_submit)
    _validate_dry_run_flags(dry_run, submit)

    config = _load_app_config(config_path, runs_dir, project_dir)
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)
    console.print(f"[bold]LabPilot[/bold] — building run [cyan]{run_id}[/cyan]\n")
    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] stopping after code generation.\n")

    pipeline = _build_pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        llm_client=llm_client,
        dry_run=dry_run,
        project_dir=project_dir,
    )
    manifest = _continue_or_exit(pipeline.build, run_id)

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Build complete:[/green] {run_dir}")
    console.print(f"[green]Reflection:[/green] {run_dir / 'reflection.md'}")
    if (run_dir / "report.html").is_file():
        console.print(f"[green]Report:[/green] {run_dir / 'report.html'}")
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


def _check_llm_or_confirm(config, assume_yes: bool) -> LLMClient | None:
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
    client = resolve_llm_client(config)
    if client is not None:
        return client

    console.print(
        "[yellow]No LLM provider available[/yellow] — brief.md/reflection.md will use "
        "template fallback text instead of AI-generated content."
    )
    for hint in llm_setup_hints(config):
        console.print(f"  [dim]•[/dim] {hint}")
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
def report(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to render"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Generate or refresh the standalone HTML report for a research run."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    run_dir = config.runs_dir / run_id
    if not (run_dir / "manifest.json").is_file():
        raise typer.BadParameter(f"Run not found: {run_id}")

    manifest = find_manifest(config, run_id)
    path = ReportGenerator().generate(run_dir, manifest)
    console.print(f"[green]Report written:[/green] {path}")


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
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    hypothesis_id: str | None = typer.Option(
        None,
        "--hypothesis",
        help="Hypothesis ID to test with this child run (e.g. H-001)",
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
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Fork, plan, and generate code without training or submission",
    ),
    assume_yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts, e.g. proceed without LLM if unavailable",
    ),
) -> None:
    """Fork a completed run, apply an improvement plan, and retrain."""
    _fail_fast_on_bad_environment(skip_lightgbm=dry_run)
    _validate_submit_flags(submit, force_submit)
    _validate_dry_run_flags(dry_run, submit)

    config = _load_app_config(config_path, runs_dir, project_dir, knowledge_dir)
    llm_client = _check_llm_or_confirm(config.llm, assume_yes)

    console.print(
        f"[bold]LabPilot[/bold] — improving run [cyan]{run_id}[/cyan] "
        f"(strategy={strategy})\n"
    )
    if dry_run:
        console.print("[yellow]Dry-run mode:[/yellow] stopping after code generation.\n")

    pipeline = _build_pipeline(
        config,
        submit=submit,
        force_submit=force_submit,
        llm_client=llm_client,
        dry_run=dry_run,
        project_dir=project_dir,
    )

    def _do_improve(_: str):
        return pipeline.improve(run_id, strategy=strategy, hypothesis_id=hypothesis_id)

    try:
        manifest = pipeline.improve(run_id, strategy=strategy, hypothesis_id=hypothesis_id)
    except FileNotFoundError as exc:
        message = str(exc)
        if "Hypothesis" in message:
            console.print(f"[red]{message}[/red]")
        else:
            console.print(f"[red]Run not found:[/red] {run_id} (check --runs-dir).")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    run_dir = config.runs_dir / manifest.run_id
    console.print(f"\n[green]Improvement complete:[/green] {run_dir}")
    console.print(f"[green]Reflection:[/green] {run_dir / 'reflection.md'}")
    console.print(
        f"\nCompare: [cyan]research runs diff --base {run_id} --compare {manifest.run_id}[/cyan]"
    )
    _print_submission_links_if_present(run_dir, submit)


def _parse_analyzer_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


@app.command()
def analyze(
    target: str = typer.Argument(
        ...,
        help="Competition slug/URL, or an analyzer name when a slug follows",
    ),
    competition: str | None = typer.Argument(
        None,
        help="Competition slug/URL (when the first argument is an analyzer name)",
    ),
    include: str | None = typer.Option(
        None, "--include", help="Comma-separated analyzers to run (e.g. papers,repositories)"
    ),
    exclude: str | None = typer.Option(
        None, "--exclude", help="Comma-separated analyzers to skip (e.g. dataset)"
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print to stdout: text or json (always writes analyze.json)",
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch sources into cache instead of reusing cached raw data"
    ),
    skip_ingest: bool = typer.Option(
        False,
        "--skip-ingest",
        help="Store analyzer artifacts but defer Knowledge Hub ingestion",
    ),
    skip_hypothesize: bool = typer.Option(
        False,
        "--skip-hypothesize",
        help="Skip Hypothesis Assistant top-N recommendations",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Analyze a competition's research landscape (writes analyze.json).

    Run every default analyzer, a single one (``research analyze papers
    <slug>``), or a subset via ``--include`` / ``--exclude``. Always persists
    ``knowledge/<slug>/research/reports/analyze.json``; ``--format`` controls
    stdout only.
    """
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")

    # First arg is an analyzer name only when a second (slug) arg is given.
    if competition is not None:
        only: str | None = target
        slug_or_url = competition
    else:
        only = None
        slug_or_url = target

    include_set = _parse_analyzer_csv(include)
    exclude_set = _parse_analyzer_csv(exclude)
    if only is not None and (include_set or exclude_set):
        raise typer.BadParameter(
            "A single analyzer argument cannot be combined with --include/--exclude."
        )

    config = _load_app_config(config_path, runs_dir, project_dir, knowledge_dir)
    context = build_context(
        slug_or_url,
        runs_dir=config.runs_dir,
        knowledge_dir=config.knowledge_dir,
        refresh=refresh,
    )

    orchestrator = AnalyzeOrchestrator(
        build_default_registry(),
        llm_client=resolve_llm_client(config.llm),
        ingest_knowledge=not skip_ingest,
        hypothesize=not skip_hypothesize,
    )
    try:
        report = orchestrator.analyze(
            context, only=only, include=include_set, exclude=exclude_set
        )
    except UnknownAnalyzerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    path = write_report(report, context.report_path)

    if output_format == "json":
        # Plain print — rich would soft-wrap and corrupt JSON meant for piping.
        print(to_json(report))
    else:
        render_terminal(report, console=console)
        console.print(f"\n[green]Wrote:[/green] {path}")


@app.command()
def ingest(
    competition: str = typer.Argument(..., help="Competition slug"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-ingest all stored artifacts even when every receipt is current",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Merge stored Layer-2 artifacts into Knowledge Units and beliefs.

    By default this is a no-op when every stored artifact has a current,
    successful Knowledge Hub receipt. If anything is new or changed, the full
    stored artifact set is merged so existing cross-source evidence is retained.
    """
    config = _load_app_config(config_path, None, project_dir, knowledge_dir)
    with KnowledgeStore(config.knowledge_dir, competition) as store:
        artifacts = store.list_artifacts()
        if not artifacts:
            console.print(
                f"[yellow]No stored artifacts found for {competition}.[/yellow] "
                "Run research analyze first."
            )
            return

        hub = KnowledgeHub(
            store,
            llm_client=resolve_llm_client(config.llm),
        )
        pending = hub.pending_artifacts(artifacts)
        if not pending and not force:
            console.print(
                f"[green]Knowledge Hub is up to date:[/green] "
                f"{len(artifacts)} artifact(s), 0 pending."
            )
            return

        result = hub.ingest(artifacts)

    console.print(
        f"[green]Knowledge ingestion complete:[/green] "
        f"{len(result.units)} unit(s), {len(result.beliefs)} belief(s) "
        f"from {len(artifacts)} artifact(s) "
        f"({len(pending)} pending{' before forced rebuild' if force else ''})."
    )
    for note in result.notes:
        console.print(f"  [yellow]•[/yellow] {note}")


@app.command("retrieve")
def retrieve_cmd(
    competition: str = typer.Argument(..., help="Competition slug"),
    question: str = typer.Option(
        "",
        "--question",
        "-q",
        help="Free-text research question (rules+optional LLM classify)",
    ),
    query_type: str = typer.Option(
        "hypothesis_generation",
        "--query-type",
        help="hypothesis_generation | structured_query | explain | compare",
    ),
    pipeline: str | None = typer.Option(
        None,
        "--pipeline",
        help="Comma-separated current techniques (e.g. EMA,Mixup,ConvNeXt)",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print: text (brief) or json (full ResearchContext)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Build typed ResearchContext from the Knowledge Store (Plan 9).

    Does not re-run analyzers. Reads ``knowledge.db`` only. Not wired into
    ``research analyze`` — Plan 10 Hypothesis Assistant consumes this API.
    """
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")
    try:
        QueryType(query_type)
    except ValueError as exc:
        raise typer.BadParameter(
            "--query-type must be one of: hypothesis_generation, "
            "structured_query, explain, compare"
        ) from exc

    config = _load_app_config(config_path, runs_dir, project_dir, knowledge_dir)
    pipeline_list = (
        [item.strip() for item in pipeline.split(",") if item.strip()] if pipeline else []
    )

    # Optional local pipeline enrichment when CLI flag omitted.
    if not pipeline_list and config.runs_dir is not None:
        try:
            from labpilot.research_engine.intelligence.context import build_context
            from labpilot.research_engine.intelligence.repositories.local_profile import (
                LocalCodeProfiler,
            )

            ctx = build_context(
                competition,
                runs_dir=config.runs_dir,
                knowledge_dir=config.knowledge_dir,
            )
            profile = LocalCodeProfiler().profile(ctx)
            if profile is not None:
                pipeline_list = list(
                    dict.fromkeys(
                        [
                            *profile.architecture,
                            *profile.augmentation,
                            *profile.training_tricks,
                            *profile.loss,
                        ]
                    )
                )
        except Exception:
            pipeline_list = []

    with KnowledgeStore(config.knowledge_dir, competition) as store:
        if not store.list_techniques(limit=1) and not store.list_artifacts():
            console.print(
                f"[yellow]No knowledge found for {competition}.[/yellow] "
                "Run research analyze + research ingest first."
            )
            raise typer.Exit(code=1)

        context = ContextBuilder(
            store,
            llm_client=resolve_llm_client(config.llm),
        ).build(
            question,
            query_type=query_type,
            pipeline=pipeline_list,
            competition={"slug": competition},
        )

    if output_format == "json":
        print(context.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]Research retrieval[/bold] — [cyan]{competition}[/cyan]")
    if context.intent is not None:
        console.print(
            f"[dim]Intent:[/dim] {context.intent.query_type} "
            f"(via {context.intent.classified_by}) "
            f"domain={context.intent.domain or '—'} "
            f"metric={context.intent.metric or '—'}"
        )
    console.print(
        f"[dim]Cards:[/dim] techniques={len(context.techniques)} "
        f"papers={len(context.papers)} experiments={len(context.experiments)} "
        f"repos={len(context.repositories)} failures={len(context.failures)}"
    )
    console.print(
        f"[dim]Budget:[/dim] {context.budget.get('total_chars', 0)}/"
        f"{context.budget.get('total_budget', 0)} chars"
    )
    console.print()
    console.print(context.brief or "(empty brief)")
    if context.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in context.notes:
            console.print(f"  [yellow]•[/yellow] {note}")


@app.command()
def hypothesize(
    competition: str = typer.Argument(..., help="Competition slug"),
    question: str = typer.Option(
        "Suggest next experiments",
        "--question",
        "-q",
        help="Framing question for ContextBuilder / drafts",
    ),
    pipeline: str | None = typer.Option(
        None,
        "--pipeline",
        help="Comma-separated current techniques (e.g. EMA,Mixup)",
    ),
    limit: int = typer.Option(10, "--limit", help="Max recommendations (≤10)"),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print: text or json",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Recommend top-N next experiments from the Knowledge Store (Plan 10).

    Recommendations only — does not train, fork, or call research improve.
    Persists Suggested hypotheses into the M2 HypothesisStore and writes
    ``research/reports/hypotheses.json``.
    """
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")
    if limit < 1 or limit > 10:
        raise typer.BadParameter("--limit must be between 1 and 10.")

    config = _load_app_config(config_path, runs_dir, project_dir, knowledge_dir)
    pipeline_list = (
        [item.strip() for item in pipeline.split(",") if item.strip()] if pipeline else []
    )
    if not pipeline_list and config.runs_dir is not None:
        try:
            from labpilot.research_engine.intelligence.context import build_context
            from labpilot.research_engine.intelligence.repositories.local_profile import (
                LocalCodeProfiler,
            )

            ctx = build_context(
                competition,
                runs_dir=config.runs_dir,
                knowledge_dir=config.knowledge_dir,
            )
            profile = LocalCodeProfiler().profile(ctx)
            if profile is not None:
                pipeline_list = list(
                    dict.fromkeys(
                        [
                            *profile.architecture,
                            *profile.augmentation,
                            *profile.training_tricks,
                            *profile.loss,
                        ]
                    )
                )
        except Exception:
            pipeline_list = []

    result = HypothesisAssistant(
        llm_client=resolve_llm_client(config.llm),
        created_by=HypothesisCreatedBy.HYPOTHESIZE,
    ).recommend(
        knowledge_dir=config.knowledge_dir,
        competition=competition,
        question=question,
        pipeline=pipeline_list,
        limit=limit,
        persist=True,
        write_report=True,
        progressive=True,
    )

    if output_format == "json":
        print(result.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]Hypothesis Assistant[/bold] — [cyan]{competition}[/cyan]")
    console.print(f"[dim]Recommendations:[/dim] {len(result.recommendations)}")
    for card in result.recommendations:
        console.print(
            f"  [cyan]#{card.rank}[/cyan] {card.title}  "
            f"[dim]({card.hypothesis_id})[/dim]"
        )
        console.print(
            f"      impact={card.expected_impact}  "
            f"confidence={card.confidence:.2f}  "
            f"effort={card.implementation_effort}  "
            f"score={card.score:.3f}"
        )
        if card.supporting_evidence:
            refs = ", ".join(f"{e.kind}:{e.ref}" for e in card.supporting_evidence[:4])
            console.print(f"      evidence: {refs}")
        if card.avoids_failure_ids:
            console.print(f"      avoids: {', '.join(card.avoids_failure_ids)}")
    if result.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in result.notes:
            console.print(f"  [yellow]•[/yellow] {note}")


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


@experiments_app.command("graph")
def experiments_graph(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    metric: str | None = typer.Option(
        None, "--metric", help="Metric key to annotate scores and highlight the best path"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Print the experiment lineage tree (parent/child relationships) for a competition."""
    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir

    graph = build_graph(config.runs_dir, competition, knowledge_dir=config.knowledge_dir)
    if not graph.nodes:
        console.print(f"No experiments found for [cyan]{competition}[/cyan] (check --runs-dir).")
        raise typer.Exit()

    console.print(f"[bold]{competition}[/bold] — {len(graph.nodes)} experiment(s)\n")
    console.print(graph.to_tree_text(metric))
    if metric:
        console.print("\n[dim]* marks the best-scoring root-to-leaf path for this metric.[/dim]")


@experiments_app.command("show")
def experiments_show(
    run_id: str = typer.Argument(..., help="Run ID to show"),
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table or json"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Show one experiment's detail view: status, progress, lineage, artifacts, metrics."""
    if output_format not in {"table", "json"}:
        raise typer.BadParameter("--format must be 'table' or 'json'.")

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir
    run_dir = config.runs_dir / run_id
    if not (run_dir / "manifest.json").is_file():
        console.print(f"[red]Run not found:[/red] {run_id} (check --runs-dir).")
        raise typer.Exit(code=1)

    manifest = load_manifest(run_dir)
    graph = build_graph(
        config.runs_dir, manifest.competition, knowledge_dir=config.knowledge_dir
    )
    experiment = graph.nodes[run_id]

    if output_format == "json":
        # Plain print, not console.print(): rich soft-wraps long lines (e.g.
        # artifact paths) at the terminal width, which corrupts JSON meant
        # for scripting/piping.
        print(experiment.model_dump_json(indent=2))
        return

    table = Table(title=f"Experiment: {experiment.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Competition", experiment.competition)
    table.add_row("Status", experiment.status)
    table.add_row("Progress", experiment.progress)
    table.add_row("Description", experiment.description)
    table.add_row("Parent", experiment.parent_id or "-")
    table.add_row("Children", ", ".join(experiment.children_ids) or "-")
    table.add_row("Iteration", str(experiment.iteration))
    table.add_row("Hypothesis", experiment.hypothesis_id or "-")
    table.add_row("Git commit", experiment.git_commit or "-")
    table.add_row("Template", experiment.template_name or "-")
    table.add_row("Problem type", experiment.problem_type or "-")
    table.add_row(
        "Metrics",
        ", ".join(f"{key}={value:.4f}" for key, value in sorted(experiment.metrics.items())) or "-",
    )
    table.add_row(
        "Public score",
        f"{experiment.public_score:.6f}" if experiment.public_score is not None else "-",
    )
    table.add_row(
        "Runtime",
        f"{experiment.runtime_seconds:.1f}s" if experiment.runtime_seconds is not None else "-",
    )
    table.add_row("Artifacts", str(len(experiment.artifacts)))
    table.add_row("Reflection", experiment.reflection_path or "-")
    table.add_row("Report", experiment.report_path or "-")
    console.print(table)


@experiments_app.command("compare")
def experiments_compare(
    base_id: str = typer.Argument(..., help="Base (parent / earlier) run ID"),
    compare_id: str = typer.Argument(..., help="Compare (child / later) run ID"),
    output_format: str = typer.Option(
        "table", "--format", help="Output format: table, json, or markdown"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Deterministic A/B comparison with categorized changes and a verdict."""
    if output_format not in {"table", "json", "markdown"}:
        raise typer.BadParameter("--format must be 'table', 'json', or 'markdown'.")

    config = load_config(config_path)
    if runs_dir:
        config.runs_dir = runs_dir

    base_dir = config.runs_dir / base_id
    compare_dir = config.runs_dir / compare_id
    if not (base_dir / "manifest.json").is_file():
        console.print(f"[red]Base run not found:[/red] {base_id} (check --runs-dir).")
        raise typer.Exit(code=1)
    if not (compare_dir / "manifest.json").is_file():
        console.print(f"[red]Compare run not found:[/red] {compare_id} (check --runs-dir).")
        raise typer.Exit(code=1)

    # Prefer on-disk comparison.json when it already records this exact pair so
    # `--format markdown` matches the persisted comparison.md byte-for-byte.
    stored = load_comparison(compare_dir)
    if stored is not None and stored.base_id == base_id and stored.compare_id == compare_id:
        comparison = stored
    else:
        base_exp = assemble_experiment(base_dir, knowledge_dir=config.knowledge_dir)
        compare_exp = assemble_experiment(compare_dir, knowledge_dir=config.knowledge_dir)
        comparator_cfg = config.experiments.comparator
        comparison = compare(
            base_exp,
            compare_exp,
            noise_epsilon=comparator_cfg.noise_epsilon,
            max_runtime_increase_pct=comparator_cfg.max_runtime_increase_pct,
            competition_dirs=(base_dir, compare_dir),
        )

    if output_format == "json":
        print(comparison.model_dump_json(indent=2))
        return

    if output_format == "markdown":
        # Plain print — same deterministic string as comparison.md on disk.
        sys.stdout.write(render_markdown(comparison))
        return

    changes_table = Table(title="Changes")
    changes_table.add_column("Category", style="cyan")
    changes_table.add_column("Change")
    if comparison.changes:
        for change in comparison.changes:
            changes_table.add_row(change.category.value, change.label)
    else:
        changes_table.add_row("-", "(no config changes detected)")
    console.print(changes_table)

    metrics_table = Table(title="Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Delta")
    for key in sorted(comparison.metric_deltas):
        delta = comparison.metric_deltas[key]
        label = f"{key} (primary)" if key == comparison.primary_metric_key else key
        metrics_table.add_row(label, f"{delta:+.4f}")
    if comparison.runtime_delta_seconds is None:
        metrics_table.add_row("Training time", "not available")
    else:
        pct = comparison.runtime_delta_pct
        pct_part = f" ({pct:+.0f}%)" if pct is not None else ""
        metrics_table.add_row(
            "Training time", f"{comparison.runtime_delta_seconds:+.1f}s{pct_part}"
        )
    metrics_table.add_row("Inference", "not tracked")
    console.print(metrics_table)

    conclusion = Table(title="Conclusion")
    conclusion.add_column("Field", style="cyan")
    conclusion.add_column("Value")
    conclusion.add_row("Verdict", comparison.verdict.value)
    conclusion.add_row("Reason", comparison.verdict_reason)
    console.print(conclusion)


@experiments_app.command("rank")
def experiments_rank(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    top: int = typer.Option(5, "--top", help="Show at most this many candidates"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Rank proposed hypotheses (recommendation backlog — does not auto-run)."""
    config = _load_app_config(config_path, runs_dir, None, knowledge_dir)
    ranking_cfg = config.experiments.ranking
    weights = RankingWeights(
        expected_gain=ranking_cfg.weights.expected_gain,
        implementation_cost=ranking_cfg.weights.implementation_cost,
        gpu_cost=ranking_cfg.weights.gpu_cost,
        risk=ranking_cfg.weights.risk,
        novelty=ranking_cfg.weights.novelty,
    )
    ranked = rank_candidates(
        competition,
        config.runs_dir,
        config.knowledge_dir,
        weights=weights,
        default_expected_gain=ranking_cfg.default_expected_gain,
        cheap_tags=set(ranking_cfg.cheap_tags),
    )
    if not ranked:
        console.print(
            f"No proposed hypotheses to rank for [cyan]{competition}[/cyan] "
            "(add some with `research hypothesis add`, or wait for reflection drafts)."
        )
        raise typer.Exit()

    table = Table(title=f"Recommended next — {competition}")
    table.add_column("#", style="cyan")
    table.add_column("ID")
    table.add_column("Prediction")
    table.add_column("Gain")
    table.add_column("Cost")
    table.add_column("GPU (s)")
    table.add_column("Risk")
    table.add_column("Novelty")
    table.add_column("Score")
    for index, candidate in enumerate(ranked[: max(1, top)], start=1):
        hyp = candidate.hypothesis
        pred = hyp.prediction if len(hyp.prediction) <= 48 else hyp.prediction[:45] + "..."
        table.add_row(
            str(index),
            hyp.id,
            pred,
            f"{candidate.expected_gain:+.4f}",
            f"{candidate.implementation_cost:.2f}",
            f"{candidate.gpu_cost_seconds:.1f}",
            f"{candidate.risk:.2f}",
            f"{candidate.novelty:.2f}",
            f"{candidate.score:.3f}",
        )
    console.print(table)
    top_candidate = ranked[0]
    console.print(
        f"\n[bold]Recommended next:[/bold] {top_candidate.hypothesis.id} — "
        f"{top_candidate.hypothesis.prediction} "
        f"(confidence {top_candidate.hypothesis.confidence:.0%}, "
        f"score {top_candidate.score:.3f})"
    )


@experiments_app.command("search")
def experiments_search(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    config_equals: list[str] = typer.Option(
        [], "--config", help="Filter key=value (repeatable), e.g. model_params.ema=true"
    ),
    recipe: list[str] = typer.Option([], "--recipe", help="Require feature recipe (repeatable)"),
    metric_gt: list[str] = typer.Option(
        [], "--metric-gt", help="Filter metric key:value greater-than (repeatable)"
    ),
    metric_lt: list[str] = typer.Option(
        [], "--metric-lt", help="Filter metric key:value less-than (repeatable)"
    ),
    metric_delta_gt: list[str] = typer.Option(
        [], "--metric-delta-gt", help="Filter comparison metric delta key:value (repeatable)"
    ),
    metric_delta_lt: list[str] = typer.Option(
        [], "--metric-delta-lt", help="Filter comparison metric delta key:value (repeatable)"
    ),
    runtime_max: str | None = typer.Option(
        None, "--runtime-max", help="Max runtime (e.g. 4h, 90m, 30s)"
    ),
    runtime_min: str | None = typer.Option(
        None, "--runtime-min", help="Min runtime (e.g. 4h, 90m, 30s)"
    ),
    verdict: str | None = typer.Option(
        None,
        "--verdict",
        help="worth_keeping|not_worth_keeping|regression|inconclusive",
    ),
    status: str | None = typer.Option(None, "--status", help="Exact experiment status"),
    template: str | None = typer.Option(None, "--template", help="Exact template_name"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config-file", help="Path to LabPilot config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Search experiments in a competition with composable AND filters."""
    # Note: --config is used for key=value filters; LabPilot config file is --config-file.
    filters = SearchFilters()
    try:
        for item in config_equals:
            filters.config_equals.append(parse_key_value(item))
        filters.recipes = list(recipe)
        for item in metric_gt:
            filters.metric_gt.append(parse_metric_threshold(item))
        for item in metric_lt:
            filters.metric_lt.append(parse_metric_threshold(item))
        for item in metric_delta_gt:
            filters.metric_delta_gt.append(parse_metric_threshold(item))
        for item in metric_delta_lt:
            filters.metric_delta_lt.append(parse_metric_threshold(item))
        if runtime_max is not None:
            filters.runtime_max_seconds = parse_duration(runtime_max)
        if runtime_min is not None:
            filters.runtime_min_seconds = parse_duration(runtime_min)
        if verdict is not None:
            filters.verdict = Verdict(verdict)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    filters.status = status
    filters.template = template

    config = _load_app_config(config_path, runs_dir, None, None)
    graph = build_graph(config.runs_dir, competition, knowledge_dir=config.knowledge_dir)
    if not graph.nodes:
        console.print(f"No experiments found for [cyan]{competition}[/cyan].")
        raise typer.Exit()

    comparisons = load_comparisons(config.runs_dir, graph)
    matches = search(graph, comparisons, filters)
    if not matches:
        console.print(f"No experiments matched filters for [cyan]{competition}[/cyan].")
        raise typer.Exit()

    table = Table(title=f"Search: {competition} ({len(matches)} match(es))")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Template")
    table.add_column("Recipes")
    table.add_column("Metrics")
    table.add_column("Runtime")
    for exp in matches:
        metrics = ", ".join(f"{k}={v:.4f}" for k, v in sorted(exp.metrics.items())[:3]) or "-"
        runtime = f"{exp.runtime_seconds:.1f}s" if exp.runtime_seconds is not None else "-"
        table.add_row(
            exp.id,
            exp.status,
            exp.template_name or "-",
            ", ".join(exp.feature_recipes) or "-",
            metrics,
            runtime,
        )
    console.print(table)


@experiments_app.command("report")
def experiments_report(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    output_format: str = typer.Option(
        "text", "--format", help="Output format: text|json"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Competition rollup: discoveries, failures, best path, recommended next."""
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be text or json")
    config = _load_app_config(config_path, runs_dir, None, knowledge_dir)
    ranking_cfg = config.experiments.ranking
    try:
        report = build_report(
            competition,
            config.runs_dir,
            config.knowledge_dir,
            weights=RankingWeights(
                expected_gain=ranking_cfg.weights.expected_gain,
                implementation_cost=ranking_cfg.weights.implementation_cost,
                gpu_cost=ranking_cfg.weights.gpu_cost,
                risk=ranking_cfg.weights.risk,
                novelty=ranking_cfg.weights.novelty,
            ),
            default_expected_gain=ranking_cfg.default_expected_gain,
            cheap_tags=set(ranking_cfg.cheap_tags),
        )
    except NoExperimentsError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit() from exc

    if output_format == "json":
        print(report.model_dump_json(indent=2))
        return
    render_report_text(report, console=console)


@experiments_app.command("dashboard")
def experiments_dashboard(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Write a static HTML competition dashboard under knowledge/<slug>/."""
    config = _load_app_config(config_path, runs_dir, None, knowledge_dir)
    ranking_cfg = config.experiments.ranking
    try:
        report = build_report(
            competition,
            config.runs_dir,
            config.knowledge_dir,
            weights=RankingWeights(
                expected_gain=ranking_cfg.weights.expected_gain,
                implementation_cost=ranking_cfg.weights.implementation_cost,
                gpu_cost=ranking_cfg.weights.gpu_cost,
                risk=ranking_cfg.weights.risk,
                novelty=ranking_cfg.weights.novelty,
            ),
            default_expected_gain=ranking_cfg.default_expected_gain,
            cheap_tags=set(ranking_cfg.cheap_tags),
        )
    except NoExperimentsError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit() from exc

    graph = build_graph(
        config.runs_dir, competition, knowledge_dir=config.knowledge_dir
    )
    path = write_dashboard(
        report,
        graph,
        knowledge_dir=config.knowledge_dir,
        runs_dir=config.runs_dir,
    )
    console.print(f"[green]Dashboard written:[/green] {path}")


@knowledge_app.command("list")
def experiments_knowledge_list(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    technique: str | None = typer.Option(
        None, "--technique", help="Filter by normalized technique name"
    ),
    effect: str | None = typer.Option(
        None,
        "--effect",
        help="Filter by effect: improves|hurts|neutral|unknown",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """List accumulated technique knowledge for a competition (Plan 5)."""
    from labpilot.experiments.models import KnowledgeEffect

    effect_filter = None
    if effect is not None:
        try:
            effect_filter = KnowledgeEffect(effect)
        except ValueError as exc:
            raise typer.BadParameter(
                "--effect must be one of: improves, hurts, neutral, unknown"
            ) from exc

    config = _load_app_config(config_path, None, None, knowledge_dir)
    kb = KnowledgeBase(config.knowledge_dir, competition)
    entries = kb.list_entries(technique=technique, effect=effect_filter)
    if not entries:
        console.print(
            f"No knowledge entries for [cyan]{competition}[/cyan] "
            f"(check --knowledge-dir / run improve to accumulate)."
        )
        raise typer.Exit()

    table = Table(title=f"Knowledge: {competition}")
    table.add_column("Technique", style="cyan")
    table.add_column("Metric")
    table.add_column("Effect")
    table.add_column("Delta")
    table.add_column("Confidence")
    table.add_column("N")
    table.add_column("Updated")
    for entry in entries:
        table.add_row(
            entry.technique,
            entry.metric_key,
            entry.effect.value,
            f"{entry.delta_estimate:+.4f}",
            f"{entry.confidence:.2f}",
            str(entry.sample_size),
            entry.updated_at.isoformat(timespec="seconds"),
        )
    console.print(table)


@hypothesis_app.command("add")
def hypothesis_add(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    observation: str = typer.Option(..., "--observation", help="What was observed"),
    reason: str = typer.Option(..., "--reason", help="Why that might be happening"),
    prediction: str = typer.Option(..., "--prediction", help="What we predict will help"),
    confidence: float = typer.Option(
        ..., "--confidence", help="Prior confidence in 0.0–1.0", min=0.0, max=1.0
    ),
    tags: str = typer.Option(
        "",
        "--tags",
        help="Comma-separated tags (e.g. loss,class-imbalance)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Create a new structured hypothesis for a competition."""
    config = _load_app_config(config_path, None, None, knowledge_dir)
    store = HypothesisStore(config.knowledge_dir, competition)
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    hypothesis = store.create(
        observation=observation,
        reason=reason,
        prediction=prediction,
        confidence=confidence,
        tags=tag_list,
        source="manual",
    )
    path = config.knowledge_dir / competition / "hypotheses" / f"{hypothesis.id}.json"
    console.print(f"[green]Created[/green] [cyan]{hypothesis.id}[/cyan] → {path}")


@hypothesis_app.command("list")
def hypothesis_list(
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    status: str | None = typer.Option(
        None, "--status", help="Filter by status: proposed, testing, confirmed, ..."
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """List hypotheses for a competition."""
    config = _load_app_config(config_path, None, None, knowledge_dir)
    store = HypothesisStore(config.knowledge_dir, competition)
    status_filter: HypothesisStatus | None = None
    if status is not None:
        try:
            status_filter = HypothesisStatus(status)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in HypothesisStatus)
            raise typer.BadParameter(f"--status must be one of: {allowed}") from exc

    hypotheses = store.list(status=status_filter)
    if not hypotheses:
        console.print(f"No hypotheses for [cyan]{competition}[/cyan].")
        raise typer.Exit()

    table = Table(title=f"Hypotheses: {competition}")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Prediction")
    for hypothesis in hypotheses:
        table.add_row(
            hypothesis.id,
            hypothesis.status.value,
            f"{hypothesis.confidence:.2f}",
            hypothesis.prediction,
        )
    console.print(table)


@hypothesis_app.command("show")
def hypothesis_show(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (e.g. H-001)"),
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Show one hypothesis and the experiments linked to it."""
    config = _load_app_config(config_path, runs_dir, None, knowledge_dir)
    store = HypothesisStore(config.knowledge_dir, competition)
    hypothesis = store.get(hypothesis_id)
    if hypothesis is None:
        console.print(
            f"[red]Hypothesis not found:[/red] {hypothesis_id} "
            f"(competition={competition})."
        )
        raise typer.Exit(code=1)

    table = Table(title=f"Hypothesis: {hypothesis.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Competition", hypothesis.competition)
    table.add_row("Status", hypothesis.status.value)
    table.add_row("Confidence", f"{hypothesis.confidence:.2f}")
    table.add_row("Observation", hypothesis.observation)
    table.add_row("Reason", hypothesis.reason)
    table.add_row("Prediction", hypothesis.prediction)
    table.add_row("Tags", ", ".join(hypothesis.tags) or "-")
    table.add_row("Source", hypothesis.source)
    table.add_row("Evidence for", ", ".join(hypothesis.evidence_for) or "-")
    table.add_row("Evidence against", ", ".join(hypothesis.evidence_against) or "-")
    console.print(table)

    graph = build_graph(
        config.runs_dir, competition, knowledge_dir=config.knowledge_dir
    )
    linked = linked_experiments(hypothesis_id, graph)
    console.print(
        f"\n[bold]Linked experiments[/bold] ({len(linked)}): "
        + (", ".join(exp.id for exp in linked) if linked else "-")
    )


@hypothesis_app.command("update")
def hypothesis_update(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (e.g. H-001)"),
    competition: str = typer.Option(..., "--competition", "-c", help="Kaggle competition slug"),
    status: str = typer.Option(..., "--status", help="New status"),
    evidence_run: str | None = typer.Option(
        None,
        "--evidence-run",
        help="Run ID supporting/contradicting the prediction (routed by --status)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Update hypothesis status (and optionally append evidence from a run)."""
    try:
        new_status = HypothesisStatus(status)
    except ValueError as exc:
        allowed = ", ".join(s.value for s in HypothesisStatus)
        raise typer.BadParameter(f"--status must be one of: {allowed}") from exc

    config = _load_app_config(config_path, None, None, knowledge_dir)
    store = HypothesisStore(config.knowledge_dir, competition)
    try:
        updated = store.update_status(
            hypothesis_id, new_status, evidence_run_id=evidence_run
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print(
        f"[green]Updated[/green] [cyan]{updated.id}[/cyan] → status={updated.status.value}"
    )
    if evidence_run and new_status == HypothesisStatus.CONFIRMED:
        console.print(f"  evidence_for += {evidence_run}")
    elif evidence_run and new_status == HypothesisStatus.REJECTED:
        console.print(f"  evidence_against += {evidence_run}")


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


@workspace_app.command("init")
def workspace_init(
    name: str = typer.Option(..., "--name", help="Project name"),
    directory: Path = typer.Option(
        Path("."), "--directory", "-d", help="Directory to initialize as a project"
    ),
) -> None:
    """Create project.yaml and standard project directories."""
    project = init_project(directory, name)
    console.print(f"[green]Project initialized:[/green] {project.root}")
    console.print(f"  Config:       {project.config_path}")
    console.print(f"  Runs:         {project.runs_dir}")
    console.print(f"  Competitions: {project.competitions_dir}")
    console.print(f"  Runtimes:     {project.runtimes_dir}")


@workspace_app.command("status")
def workspace_status(
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
) -> None:
    """Show resolved project paths and run counts."""
    project = load_project(project_dir=project_dir)
    if project is None:
        console.print("[yellow]No project.yaml found[/yellow] in cwd or parents.")
        raise typer.Exit(code=1)

    runs_count = 0
    if project.runs_dir.is_dir():
        runs_count = sum(1 for path in project.runs_dir.iterdir() if path.is_dir())

    table = Table(title=f"Project: {project.name}")
    table.add_column("Setting", style="cyan")
    table.add_column("Path")
    table.add_row("Root", str(project.root))
    table.add_row("Config", str(project.config_path))
    table.add_row("Runs dir", str(project.runs_dir))
    table.add_row("Competitions dir", str(project.competitions_dir))
    table.add_row("Runtimes dir", str(project.runtimes_dir))
    table.add_row("Default runtime", project.default_runtime)
    table.add_row("Runs", str(runs_count))
    console.print(table)


@runtime_app.command("list")
def runtime_list(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Show only enabled runtimes"),
) -> None:
    """List registered training runtimes."""
    config = load_config(config_path, project_dir=project_dir)
    runtimes_dir = resolve_runtimes_dir(config, project_dir=project_dir)
    runtimes = list_runtimes(runtimes_dir, enabled_only=enabled_only)

    table = Table(title="Runtimes")
    table.add_column("ID", style="cyan")
    table.add_column("Provider")
    table.add_column("Enabled")
    table.add_column("Priority")
    table.add_column("Labels")
    for runtime in runtimes:
        table.add_row(
            runtime.id,
            runtime.provider,
            str(runtime.enabled),
            str(runtime.priority),
            ", ".join(runtime.labels),
        )
    console.print(table)


@runtime_app.command("show")
def runtime_show(
    runtime_id: str = typer.Option(..., "--runtime", help="Runtime ID to display"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
) -> None:
    """Show a runtime configuration (secrets redacted)."""
    config = load_config(config_path, project_dir=project_dir)
    runtimes_dir = resolve_runtimes_dir(config, project_dir=project_dir)
    runtime = get_runtime(runtime_id, runtimes_dir)
    if runtime is None:
        console.print(f"[red]Runtime not found:[/red] {runtime_id}")
        raise typer.Exit(code=1)
    console.print(yaml.safe_dump(runtime_to_yaml_dict(runtime), sort_keys=False))


@runtime_app.command("register")
def runtime_register(
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Runtime provider: local, kaggle_kernel, google_colab, or other",
    ),
    runtime_id: str | None = typer.Option(None, "--id", help="Runtime ID (defaults to provider name)"),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
) -> None:
    """Register a runtime by writing a scaffold YAML file."""
    valid_providers = {"local", "kaggle_kernel", "google_colab", "other"}
    if provider not in valid_providers:
        raise typer.BadParameter(f"provider must be one of: {', '.join(sorted(valid_providers))}")

    config = load_config(config_path, project_dir=project_dir)
    runtimes_dir = resolve_runtimes_dir(config, project_dir=project_dir)
    runtimes_dir.mkdir(parents=True, exist_ok=True)

    resolved_id = runtime_id or provider.replace("_", "-")
    output = runtimes_dir / f"{resolved_id}.yaml"
    if output.exists():
        console.print(f"[red]Runtime already exists:[/red] {output}")
        raise typer.Exit(code=1)

    runtime = scaffold_runtime(provider, resolved_id)
    output.write_text(yaml.safe_dump(runtime_to_yaml_dict(runtime), sort_keys=False))
    console.print(f"[green]Registered runtime:[/green] {output}")


@runtime_app.command("doctor")
def runtime_doctor(
    runtime_id: str | None = typer.Option(
        None, "--runtime", help="Check a single runtime (default: all registered)"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Project root containing project.yaml"
    ),
) -> None:
    """Validate runtime credentials and configuration."""
    config = load_config(config_path, project_dir=project_dir)
    runtimes_dir = resolve_runtimes_dir(config, project_dir=project_dir)
    try:
        results = check_all_runtimes(
            runtimes_dir,
            runtime_id=runtime_id,
            kaggle_username=config.kaggle.username,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    all_ok = True
    for result in results:
        console.print(f"\n[bold]{result.runtime_id}[/bold] ({result.provider})")
        for check in result.checks:
            style = "green" if check.ok else "red"
            console.print(f"  [{style}]{'✔' if check.ok else '✘'}[/{style}] {check.name}: {check.detail}")
            if not check.ok and check.fix:
                console.print(f"      [dim]{check.fix}[/dim]")
        if not result.ok:
            all_ok = False

    if not all_ok:
        raise typer.Exit(code=1)


@app.command("templates")
def templates_list() -> None:
    """List registered baseline templates."""
    table = Table(title="Baseline templates")
    table.add_column("Name", style="cyan")
    table.add_column("Problem type")
    table.add_column("Model family")
    table.add_column("Description")
    for template in list_templates():
        table.add_row(
            template.name,
            template.problem_type,
            template.model_family,
            template.description,
        )
    console.print(table)


if __name__ == "__main__":
    app()
