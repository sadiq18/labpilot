import logging
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from typer.core import TyperGroup

from labpilot.research_engine.execution.baseline.registry import list_templates
from labpilot.cli.config_helpers import (
    default_tools,
    load_cli_config,
    resolve_competition,
    resolve_os_workspace,
)
from labpilot.cli.plan import plan_app
from labpilot.cli.reflect import claims_app, reflect_app
from labpilot.config import (
    AppConfig,
    load_config,
    resolve_runtimes_dir,
)
from labpilot.workspace import discover_workspace
from labpilot.diagnostics import (
    check_environment,
    print_diagnostics_report,
    required_environment_checks,
)
from labpilot.research_engine.shared.experiments.comparator import compare, load_comparison, render_markdown
from labpilot.research_engine.shared.experiments.graph import assemble_experiment, build_graph
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore, linked_experiments
from labpilot.research_engine.shared.experiments.knowledge import KnowledgeBase
from labpilot.research_engine.shared.experiments.models import HypothesisCreatedBy, HypothesisStatus, Verdict
from labpilot.research_engine.shared.experiments.ranking import RankingWeights, rank_candidates
from labpilot.research_engine.shared.experiments.report import (
    NoExperimentsError,
    build_report,
    render_report_text,
    write_dashboard,
)
from labpilot.research_engine.shared.experiments.search import (
    SearchFilters,
    load_comparisons,
    parse_duration,
    parse_key_value,
    parse_metric_threshold,
    search,
)
from labpilot.accessor.kaggle.client import SubmissionResult
from labpilot.llm.client import LLMClient, llm_setup_hints, resolve_llm_client
from labpilot.research_engine.shared.experiments.manifest import StageStatus, find_manifest, load_manifest
from labpilot.research_engine.intelligence.fetch import KaggleFetchService
from labpilot.research_engine.intelligence.hypothesis import HypothesisAssistant
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.registry import UnknownAnalyzerError
from labpilot.research_engine.intelligence.renderers.json import to_json
from labpilot.research_engine.intelligence.renderers.terminal import render_terminal
from labpilot.research_engine.intelligence.retrieval import (
    ContextBuilder,
    QueryType,
)
from labpilot.research_engine.execution.runtimes.doctor import check_all_runtimes
from labpilot.research_engine.execution.runtimes.registry import get_runtime, list_runtimes
from labpilot.research_engine.execution.runtimes.templates import runtime_to_yaml_dict, scaffold_runtime
from labpilot.research_engine.shared.experiments.index import diff_runs

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = typer.Typer(
    name="research",
    help="LabPilot — one-command Kaggle competition research engine",
    no_args_is_help=True,
)
runs_app = typer.Typer(help="Inspect and compare research runs.")
app.add_typer(runs_app, name="runs")
runtime_app = typer.Typer(help="Register and validate training runtimes.")
app.add_typer(runtime_app, name="runtime")
experiments_app = typer.Typer(help="Explore the experiment graph.")
app.add_typer(experiments_app, name="experiments")
knowledge_app = typer.Typer(help="Explore the accumulated experiment knowledge base.")
experiments_app.add_typer(knowledge_app, name="knowledge")


class _HypothesizeGroup(TyperGroup):
    """Treat ``research hypothesize <slug>`` as ``research hypothesize new <slug>``.

    Bare ``research hypothesize`` (or flags only) becomes ``new`` so a workspace
    can omit the slug entirely.
    """

    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        if not args:
            args = ["new"]
        elif args[0] not in self.commands:
            # Keep group-level --help so subcommands stay visible.
            if args[0] not in {"--help", "-h"}:
                args = ["new", *args]
        return super().parse_args(ctx, args)


hypothesize_app = typer.Typer(
    cls=_HypothesizeGroup,
    help="Generate, inspect, and update hypotheses.",
    no_args_is_help=True,
)
app.add_typer(hypothesize_app, name="hypothesize")
app.add_typer(plan_app, name="plan")
app.add_typer(reflect_app, name="reflect")
app.add_typer(claims_app, name="claims")
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
    knowledge_dir: Path | None = None,
    workspace_path: Path | None = None,
) -> AppConfig:
    config, _ = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
        workspace_path=workspace_path,
    )
    return config


def _load_config_and_competition(
    competition: str | None,
    config_path: Path,
    runs_dir: Path | None = None,
    knowledge_dir: Path | None = None,
    workspace_path: Path | None = None,
) -> tuple[AppConfig, str]:
    """Load config with workspace discovery and resolve competition slug."""
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
        workspace_path=workspace_path,
    )
    return config, resolve_competition(competition, workspace)


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
    plan_id: str | None = typer.Option(
        None,
        "--plan",
        "-p",
        help="Research plan id to execute (e.g. P-001). Required.",
    ),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None,
        "--workspace",
        help="Competition workspace root (directory with labpilot.yaml)",
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Allow Kaggle upload from submission capability (default: package only)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Syntax/smoke stub path without full training or upload",
    ),
    install_packages: bool = typer.Option(
        True,
        "--install-packages/--no-install-packages",
        help="Whether Dependency capability may run pip install",
    ),
    # Legacy flags retained only so old scripts get a clear migration error.
    runs_dir: Path | None = typer.Option(None, "--runs-dir", hidden=True),
    competitions_dir: Path | None = typer.Option(None, "--competitions-dir", hidden=True),
    hypothesis_id: str | None = typer.Option(None, "--hypothesis", hidden=True),
    force_submit: bool = typer.Option(False, "--force-submit", hidden=True),
    assume_yes: bool = typer.Option(False, "--yes", "-y", hidden=True),
) -> None:
    """Run an approved ResearchPlan via the Research Engineer.

    Requires ``--plan P-xxx``. Competition defaults from ``labpilot.yaml`` when
    present. Create a plan first::

        research plan create --baseline
        research run --plan P-001
    """
    if plan_id is None:
        console.print(
            "[red]Plan-driven run required.[/red] Create a plan first, then:\n"
            "  research plan create --baseline\n"
            "  research run --plan P-001\n\n"
            "Inside a competition workspace (``labpilot.yaml``), ``--competition`` "
            "is optional. The linear Pipeline path without ``--plan`` is retired."
        )
        raise typer.Exit(code=1)

    from labpilot.cli.run_engineer import run_plan_command

    run_plan_command(
        plan_id=plan_id,
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        dry_run=dry_run,
        submit=submit or force_submit,
        install_packages=install_packages,
        workspace_path=workspace_path,
    )


@app.command("submit")
def submit_cmd(
    execution_id: str = typer.Option(
        ...,
        "--execution",
        "-e",
        help="Execution id whose submission_<E-id>.csv to upload (e.g. E-001)",
    ),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None,
        "--workspace",
        help="Competition workspace root (directory with labpilot.yaml)",
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Override CSV path (default: artifacts/submission_<execution>.csv)",
    ),
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Kaggle submission message",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resolve file and patch paths without uploading",
    ),
) -> None:
    """Upload an execution-scoped submission and record leaderboard learning.

    After a successful ``research run``, package lives at
    ``artifacts/submission_E-xxx.csv``. This command uploads that file, stores
    ``public_score`` on the linked hypothesis, updates beliefs/techniques/claims,
    and notifies proposed hypotheses. Mints a new hypothesis only when it is an
    improvement fork with expected gain (e.g. overfit → regularization).
    """
    from labpilot.cli.submit import submit_command

    submit_command(
        execution_id=execution_id,
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
        path=path,
        message=message,
        dry_run=dry_run,
    )


@app.command()
def resume(
    execution_id: str | None = typer.Option(
        None,
        "--execution",
        "-e",
        help="Execution id to resume (e.g. E-001)",
    ),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None,
        "--workspace",
        help="Competition workspace root (directory with labpilot.yaml)",
    ),
    submit: bool = typer.Option(
        False,
        "--submit",
        help="Allow Kaggle upload from submission capability",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Resume with dry-run capability constraints",
    ),
    install_packages: bool = typer.Option(
        True,
        "--install-packages/--no-install-packages",
        help="Whether Dependency capability may run pip install",
    ),
) -> None:
    """Resume a Research Engineer execution (``E-xxx``)."""
    if execution_id is None:
        console.print(
            "[red]Provide --execution E-xxx "
            "(and --competition <slug> unless inside a workspace).[/red]"
        )
        raise typer.Exit(code=1)

    from labpilot.cli.run_engineer import resume_execution_command

    resume_execution_command(
        execution_id=execution_id,
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        dry_run=dry_run,
        submit=submit,
        install_packages=install_packages,
        workspace_path=workspace_path,
    )



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
    workspace = discover_workspace()
    if workspace is not None:
        console.print(
            f"[bold]Competition workspace[/bold]  "
            f"[cyan]{workspace.root}[/cyan]  "
            f"(slug=[cyan]{workspace.competition}[/cyan])"
        )
    else:
        console.print(
            "[dim]No labpilot.yaml above CWD — using legacy CWD knowledge/ + "
            "competitions/ paths. Prefer `research init <slug> --path <root>`.[/dim]"
        )
    results = check_environment()
    all_ok = print_diagnostics_report(results, console)
    if not all_ok:
        raise typer.Exit(code=1)


@app.command("init")
def init_cmd(
    competition: str = typer.Argument(
        ...,
        help="Competition slug or Kaggle URL",
    ),
    path: Path = typer.Option(
        ...,
        "--path",
        "-p",
        help="Parent directory; creates <path>/<slug>/",
    ),
    git: bool = typer.Option(False, "--git", help="Initialize a git repo (no prompt)"),
    no_git: bool = typer.Option(
        False, "--no-git", help="Skip git without prompting"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Allow scaffolding into a non-empty directory",
    ),
) -> None:
    """Scaffold a client-owned competition workspace under ``<path>/<slug>/``.

    Writes ``labpilot.yaml``, dirs (knowledge, pipeline, data, …), ``.gitignore``,
    and an optional git commit. Does **not** download data or run analyze.

    After init, ``cd`` into the workspace and run commands with CWD discovery::

        uv run --project /path/to/labpilot research analyze
        uv run --project /path/to/labpilot research plan create --baseline
    """
    from labpilot.cli.init_workspace import init_workspace_command

    git_choice: bool | None
    if git and no_git:
        raise typer.BadParameter("Use either --git or --no-git, not both.")
    if git:
        git_choice = True
    elif no_git:
        git_choice = False
    else:
        git_choice = None

    init_workspace_command(
        competition=competition,
        path=path,
        git=git_choice,
        force=force,
        labpilot_hint=Path.cwd(),
    )


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


@app.command("journal")
def journal(
    competition: str | None = typer.Option(None, "--competition", "-c"),
    output_json: bool = typer.Option(False, "--json"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    """Print the research journal (evidence tiers, beliefs, claims, next step)."""
    from labpilot.cli.reflect import journal_cmd

    journal_cmd(
        competition=competition,
        output_json=output_json,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Legacy run id (removed)"),
) -> None:
    """Removed — use ``research journal --competition <slug>`` instead."""
    console.print(
        "[red]research report was removed[/red] (Pipeline-era HTML). "
        "Use: [cyan]research journal --competition <slug>[/cyan]"
    )
    raise typer.Exit(code=1)


def _parse_analyzer_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


@app.command()
def analyze(
    target: str | None = typer.Argument(
        None,
        help="Competition slug/URL, or an analyzer name when a slug follows "
        "(optional inside a labpilot.yaml workspace)",
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
        help="Store analyzer artifacts but defer Knowledge Hub ingestion (also skips hypotheses)",
    ),
    skip_hypothesize: bool = typer.Option(
        False,
        "--skip-hypothesize",
        help="Skip generating new hypotheses after ingestion",
    ),
    skip_brief: bool = typer.Option(
        False,
        "--skip-brief",
        help="Skip writing the Research Brief (requires ingest + hypothesize)",
    ),
    fetch_kaggle: bool = typer.Option(
        False,
        "--fetch-kaggle",
        help=(
            "Also pull Kaggle kernels (5 by votes + 5 by score) and discussions (5), "
            "then ingest/hypothesize/brief over that evidence"
        ),
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None,
        "--workspace",
        help="Competition workspace root (directory with labpilot.yaml)",
    ),
) -> None:
    """Understand the problem: artifacts, beliefs, hypotheses, and Research Brief.

    Runs default analyzers (or a subset), persists competition/dataset/research
    artifacts into ``knowledge.db``, ingests beliefs, generates new hypotheses,
    and writes ``analyze.json`` plus ``research_brief.md``.
    """
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")

    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
        workspace_path=workspace_path,
    )

    # First arg is an analyzer name only when a second (slug) arg is given.
    if competition is not None:
        only: str | None = target
        slug_or_url = competition
    elif target is not None:
        only = None
        slug_or_url = target
    else:
        only = None
        slug_or_url = resolve_competition(None, client)

    from labpilot.research_engine.intelligence.context import normalize_competition

    try:
        slug, competition_url = normalize_competition(slug_or_url)
    except ValueError:
        slug, competition_url = slug_or_url, None

    if client is not None and (competition is not None or target is not None):
        resolve_competition(slug, client, required=False)

    include_set = _parse_analyzer_csv(include)
    exclude_set = _parse_analyzer_csv(exclude)
    if only is not None and (include_set or exclude_set):
        raise typer.BadParameter(
            "A single analyzer argument cannot be combined with --include/--exclude."
        )

    ws = resolve_os_workspace(
        competition=slug,
        config=config,
        client=client,
        runs_dir=runs_dir,
    )
    do_ingest = not skip_ingest
    do_hypothesize = not skip_hypothesize and do_ingest
    do_brief = not skip_brief and do_ingest and do_hypothesize
    try:
        result = default_tools().invoke(
            "analyze_competition",
            ws,
            only=only,
            include=include_set,
            exclude=exclude_set,
            llm_client=resolve_llm_client(config.llm),
            ingest_knowledge=do_ingest,
            hypothesize=do_hypothesize,
            brief=do_brief,
            fetch_kaggle=fetch_kaggle,
            refresh=refresh,
            url=competition_url or slug_or_url,
        )
    except UnknownAnalyzerError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    report = result.data["report"]
    path = Path(result.data["path"]) if result.data.get("path") else None
    brief_path = None
    if do_brief and getattr(report, "research_brief", None):
        from labpilot.research_engine.intelligence.brief.models import ResearchBrief
        from labpilot.research_engine.intelligence.renderers.markdown import write_brief

        brief_path = write_brief(
            ResearchBrief.model_validate(report.research_brief),
            Path(result.data["brief_path"]),
        )

    if output_format == "json":
        # Plain print — rich would soft-wrap and corrupt JSON meant for piping.
        print(to_json(report))
    else:
        render_terminal(report, console=console)
        if path is not None:
            console.print(f"\n[green]Wrote:[/green] {path}")
        if brief_path is not None:
            console.print(f"[green]Research Brief:[/green] {brief_path}")


@app.command()
def ingest(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-ingest all stored artifacts even when every receipt is current",
    ),
    skip_hypothesize: bool = typer.Option(
        False,
        "--skip-hypothesize",
        help="Skip generating new hypotheses after ingestion",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Merge stored Layer-2 artifacts into Knowledge Units and beliefs.

    Hub merging is a no-op when every stored artifact has a current, successful
    receipt. If anything is new or changed, the full stored artifact set is
    merged so existing cross-source evidence is retained. New hypotheses are
    generated afterwards unless ``--skip-hypothesize`` is passed.
    """
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, workspace)
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
        else:
            result = hub.ingest(artifacts)
            console.print(
                f"[green]Knowledge ingestion complete:[/green] "
                f"{len(result.units)} unit(s), {len(result.beliefs)} belief(s) "
                f"from {len(artifacts)} artifact(s) "
                f"({len(pending)} pending{' before forced rebuild' if force else ''})."
            )
            for note in result.notes:
                console.print(f"  [yellow]•[/yellow] {note}")

    if skip_hypothesize:
        console.print("[dim]Hypothesis generation skipped by request.[/dim]")
        return

    hypotheses = _generate_hypotheses(competition, config)
    console.print(f"[green]{hypotheses.new_count} new hypothesis generated.[/green]")
    for card in hypotheses.recommendations:
        console.print(f"  [cyan]{card.hypothesis_id}[/cyan] {card.title}")


@app.command("retrieve")
def retrieve_cmd(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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

    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    competition = resolve_competition(competition, workspace)
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


def _profile_pipeline(competition: str, config: AppConfig) -> list[str]:
    """Best-effort current-pipeline techniques from local code (empty on failure)."""
    if config.runs_dir is None:
        return []
    try:
        from labpilot.research_engine.intelligence.repositories.local_profile import (
            LocalCodeProfiler,
        )

        ctx = build_context(
            competition,
            runs_dir=config.runs_dir,
            knowledge_dir=config.knowledge_dir,
        )
        profile = LocalCodeProfiler().profile(ctx)
        if profile is None:
            return []
        return list(
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
        return []


def _generate_hypotheses(
    competition: str,
    config: AppConfig,
    *,
    question: str = "Suggest next experiments",
    pipeline: list[str] | None = None,
    limit: int = 10,
    write_report: bool = True,
):
    """Run the Hypothesis Assistant, persisting only newly generated hypotheses."""
    return HypothesisAssistant(
        llm_client=resolve_llm_client(config.llm),
        created_by=HypothesisCreatedBy.HYPOTHESIZE,
    ).recommend(
        knowledge_dir=config.knowledge_dir,
        competition=competition,
        question=question,
        pipeline=pipeline if pipeline is not None else _profile_pipeline(competition, config),
        limit=limit,
        persist=True,
        write_report=write_report,
        progressive=True,
    )


@hypothesize_app.command("new")
def hypothesize_new(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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
    limit: int = typer.Option(10, "--limit", help="Max new hypotheses (≤10)"),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print: text or json",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Generate new hypotheses from the Knowledge Store (Plan 10).

    Recommendations only — does not train, fork, or call research improve.
    Techniques already tried or already covered by an open hypothesis are
    skipped, so re-running only adds genuinely new hypotheses.
    """
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("--format must be 'text' or 'json'.")
    if limit < 1 or limit > 10:
        raise typer.BadParameter("--limit must be between 1 and 10.")

    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        runs_dir=runs_dir,
    )
    competition = resolve_competition(competition, workspace)
    pipeline_list = (
        [item.strip() for item in pipeline.split(",") if item.strip()] if pipeline else None
    )
    result = _generate_hypotheses(
        competition,
        config,
        question=question,
        pipeline=pipeline_list,
        limit=limit,
    )

    if output_format == "json":
        print(result.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]Hypothesis Assistant[/bold] — [cyan]{competition}[/cyan]")
    console.print(f"[green]{result.new_count} new hypothesis generated.[/green]")
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


@app.command("fetch")
def fetch_cmd(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    source: str = typer.Option(
        "all",
        "--source",
        help="What to pull: discussions, kernels, or all",
    ),
    sort: str = typer.Option(
        "votes",
        "--sort",
        help="kernels: votes|score; discussions always use votes→top",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        help="Unique NEW artifacts to store per selected source",
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Re-pull and overwrite existing artifacts / raw versions",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Fetch Kaggle kernels and/or discussions into the research store.

    Uses the official Kaggle API (no HTML scrape). Stores
    ``ResearchArtifact`` rows (kernels as ``repository``/``kaggle``,
    discussions as ``discussion``/``kaggle``) plus RawStore blobs.
    ``--limit`` counts newly written unique ids only — pages until the
    unique count is met. Micro Agents enrich when an LLM is configured.
    """
    source_key = source.strip().lower()
    if source_key not in {"discussions", "kernels", "all"}:
        raise typer.BadParameter("--source must be discussions, kernels, or all.")
    sort_key = sort.strip().lower()
    if sort_key not in {"votes", "score"}:
        raise typer.BadParameter("--sort must be votes or score.")
    if limit < 1:
        raise typer.BadParameter("--limit must be >= 1.")

    sources: set[str]
    if source_key == "all":
        sources = {"discussions", "kernels"}
    else:
        sources = {source_key}

    kernel_sort = "scoreDescending" if sort_key == "score" else "voteCount"
    # Discussions: UI votes ↔ API top (score sort does not apply).
    discussion_sort = "top"

    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, workspace)
    from labpilot.accessor.kaggle.client import KaggleClient

    service = KaggleFetchService(
        llm_client=resolve_llm_client(config.llm),
        kaggle=KaggleClient(config.kaggle),
    )
    result = service.fetch(
        competition,
        sources=sources,  # type: ignore[arg-type]
        kernel_sort=kernel_sort,  # type: ignore[arg-type]
        discussion_sort=discussion_sort,  # type: ignore[arg-type]
        limit=limit,
        refresh=refresh,
        knowledge_dir=config.knowledge_dir,
    )

    console.print(f"\n[bold]Kaggle fetch[/bold] — [cyan]{competition}[/cyan]")
    console.print(f"[dim]Sources:[/dim] {', '.join(result.sources)}")
    console.print(
        f"  fetched={result.fetched}  skipped_existing={result.skipped_existing}  "
        f"written={result.written}  pages={result.pages_scanned}"
    )
    console.print(
        f"  enriched llm={result.llm_enriched}  "
        f"rule_engine={result.rule_engine_enriched}"
    )
    if result.artifact_ids:
        console.print("\n[bold]Wrote[/bold]")
        for artifact_id in result.artifact_ids[:20]:
            console.print(f"  • {artifact_id}")
        if len(result.artifact_ids) > 20:
            console.print(f"  [dim]… +{len(result.artifact_ids) - 20} more[/dim]")
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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    metric: str | None = typer.Option(
        None, "--metric", help="Metric key to annotate scores and highlight the best path"
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Print the experiment lineage tree (parent/child relationships) for a competition."""
    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir
    )

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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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
    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir, knowledge_dir=knowledge_dir
    )
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
            "(generate some with `research hypothesize`, or wait for reflection drafts)."
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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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

    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir
    )
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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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
    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir, knowledge_dir=knowledge_dir
    )
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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
) -> None:
    """Write a static HTML competition dashboard under knowledge/<slug>/."""
    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir, knowledge_dir=knowledge_dir
    )
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
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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
    from labpilot.research_engine.shared.experiments.models import KnowledgeEffect

    effect_filter = None
    if effect is not None:
        try:
            effect_filter = KnowledgeEffect(effect)
        except ValueError as exc:
            raise typer.BadParameter(
                "--effect must be one of: improves, hurts, neutral, unknown"
            ) from exc

    config, competition = _load_config_and_competition(
        competition, config_path, knowledge_dir=knowledge_dir
    )
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


def _hypothesis_parent_label(hypothesis) -> str:
    """Human-readable parent / fork lineage for list/show."""
    parent = getattr(hypothesis, "parent_hypothesis_id", None)
    if parent:
        return f"fork:{parent}"
    for tag in getattr(hypothesis, "tags", []) or []:
        text = str(tag)
        if text.lower().startswith("fork:"):
            return text
    return "—"


def _hypothesis_technique_label(hypothesis) -> str:
    combo = list(getattr(hypothesis, "combo_techniques", None) or [])
    if combo:
        return " + ".join(combo)
    technique = getattr(hypothesis, "technique", None)
    if technique:
        return str(technique)
    return "—"


@hypothesize_app.command("list")
def hypothesis_list(
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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
    config, competition = _load_config_and_competition(
        competition, config_path, knowledge_dir=knowledge_dir
    )
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
    table.add_column("Parent / fork")
    table.add_column("Technique")
    table.add_column("Confidence")
    table.add_column("Prediction")
    for hypothesis in hypotheses:
        table.add_row(
            hypothesis.id,
            hypothesis.status.value,
            _hypothesis_parent_label(hypothesis),
            _hypothesis_technique_label(hypothesis),
            f"{hypothesis.confidence:.2f}",
            hypothesis.prediction,
        )
    console.print(table)


@hypothesize_app.command("show")
def hypothesis_show(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (e.g. H-001)"),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    runs_dir: Path | None = typer.Option(None, "--runs-dir", help="Override runs directory"),
) -> None:
    """Show one hypothesis and the experiments linked to it."""
    config, competition = _load_config_and_competition(
        competition, config_path, runs_dir=runs_dir, knowledge_dir=knowledge_dir
    )
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
    table.add_row("Parent / fork", _hypothesis_parent_label(hypothesis))
    table.add_row("Technique", _hypothesis_technique_label(hypothesis))
    table.add_row(
        "Technique stack",
        " → ".join(hypothesis.technique_stack) if hypothesis.technique_stack else "-",
    )
    table.add_row(
        "Combo techniques",
        " + ".join(hypothesis.combo_techniques)
        if hypothesis.combo_techniques
        else "-",
    )
    table.add_row("Confidence", f"{hypothesis.confidence:.2f}")
    table.add_row(
        "Expected impact",
        f"{hypothesis.expected_impact:+.4f}" if hypothesis.expected_impact else "-",
    )
    table.add_row("Observation", hypothesis.observation)
    table.add_row("Reason", hypothesis.reason)
    table.add_row("Prediction", hypothesis.prediction)
    table.add_row("Tags", ", ".join(hypothesis.tags) or "-")
    table.add_row("Source", hypothesis.source)
    table.add_row("Evidence for", ", ".join(hypothesis.evidence_for) or "-")
    table.add_row("Evidence against", ", ".join(hypothesis.evidence_against) or "-")
    console.print(table)

    try:
        from labpilot.research_engine.evidence.store import EvidenceCardStore

        card = EvidenceCardStore(config.knowledge_dir, competition).get_for_hypothesis(
            hypothesis_id
        )
        if card is not None:
            ev = Table(title=f"Evidence Card: {card.id}")
            ev.add_column("Field", style="cyan")
            ev.add_column("Value")
            ev.add_row("Control", card.control_experiment or "—")
            ev.add_row("Treatment", card.treatment_experiment)
            ev.add_row("Decision", card.decision.value)
            ev.add_row(
                "Expected cv_gain",
                f"{card.expected.cv_gain:+.6g}"
                if card.expected.cv_gain is not None
                else "—",
            )
            ev.add_row(
                "Observed cv_gain",
                f"{card.observed.cv_gain:+.6g}"
                if card.observed.cv_gain is not None
                else "—",
            )
            ev.add_row(
                "Observed lb_gain",
                f"{card.observed.lb_gain:+.6g}"
                if card.observed.lb_gain is not None
                else "—",
            )
            ev.add_row("Stability", card.observed.stability.value)
            ev.add_row(
                "Impact error",
                f"{card.impact_error:+.6g}" if card.impact_error is not None else "—",
            )
            attrib = ", ".join(
                f"{k}={v:+.4g}" for k, v in card.technique_attribution.items()
            )
            ev.add_row("Attribution", attrib or "—")
            ev.add_row("Reusable for", ", ".join(card.reusable_for) or "—")
            console.print(ev)
    except Exception:
        pass

    graph = build_graph(
        config.runs_dir, competition, knowledge_dir=config.knowledge_dir
    )
    linked = linked_experiments(hypothesis_id, graph)
    console.print(
        f"\n[bold]Linked experiments[/bold] ({len(linked)}): "
        + (", ".join(exp.id for exp in linked) if linked else "-")
    )


@hypothesize_app.command("update")
def hypothesis_update(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID (e.g. H-001)"),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
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

    config, competition = _load_config_and_competition(
        competition, config_path, knowledge_dir=knowledge_dir
    )
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



@runtime_app.command("list")
def runtime_list(
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Show only enabled runtimes"),
) -> None:
    """List registered training runtimes."""
    config = load_config(config_path)
    runtimes_dir = resolve_runtimes_dir(config)
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
) -> None:
    """Show a runtime configuration (secrets redacted)."""
    config = load_config(config_path)
    runtimes_dir = resolve_runtimes_dir(config)
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
) -> None:
    """Register a runtime by writing a scaffold YAML file."""
    valid_providers = {"local", "kaggle_kernel", "google_colab", "other"}
    if provider not in valid_providers:
        raise typer.BadParameter(f"provider must be one of: {', '.join(sorted(valid_providers))}")

    config = load_config(config_path)
    runtimes_dir = resolve_runtimes_dir(config)
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
) -> None:
    """Validate runtime credentials and configuration."""
    config = load_config(config_path)
    runtimes_dir = resolve_runtimes_dir(config)
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
