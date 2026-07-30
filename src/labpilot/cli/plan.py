"""CLI for Research Planner — plan-only operations (no execution).

```bash
research plan create <competition> --hypothesis H-xxx
research plan create --baseline          # slug from labpilot.yaml
research plan show <competition> <plan-id>
research plan list <competition>
```
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner import (
    BaselinePlanError,
    compile_baseline_plan,
    compile_research_plan,
)
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus
from labpilot.research_engine.planner.serializer import render_markdown
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.planner.validator import topological_levels

plan_app = typer.Typer(
    help="Compile and inspect research plans (plan-only; never executes tasks).",
    no_args_is_help=True,
)
console = Console()

_FORMATS = ("text", "json", "markdown")


def _validate_format(value: str) -> str:
    key = value.strip().lower()
    if key not in _FORMATS:
        raise typer.BadParameter(f"--format must be one of: {', '.join(_FORMATS)}")
    return key


def _parse_status(value: str | None) -> PlanStatus | None:
    if value is None:
        return None
    try:
        return PlanStatus(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(s.value for s in PlanStatus)
        raise typer.BadParameter(f"--status must be one of: {allowed}") from exc


def _print_plan(plan: ResearchPlan, output_format: str) -> None:
    if output_format == "json":
        print(plan.model_dump_json(indent=2))
        return
    if output_format == "markdown":
        print(render_markdown(plan), end="")
        return

    kind = plan.metadata.get("plan_kind") or (
        "hypothesis" if plan.hypothesis_id else "—"
    )
    console.print(f"\n[bold]Research Plan[/bold] [cyan]{plan.id}[/cyan]")
    console.print(f"  kind: {kind}")
    console.print(f"  hypothesis: {plan.hypothesis_id or '—'}")
    console.print(f"  status: {plan.status}  generated_by: {plan.generated_by}")
    console.print(f"  priority: {plan.priority}  gain: {plan.estimated_gain}")
    console.print(f"  goal: {plan.goal or '—'}")
    if plan.risk:
        console.print(f"  risk: {plan.risk}")
    if plan.success_criteria:
        console.print("  success_criteria:")
        for item in plan.success_criteria:
            console.print(f"    • {item}")

    console.print("\n[bold]Task DAG[/bold] (topological levels)")
    levels = topological_levels(plan)
    task_by_id = {task.id: task for task in plan.tasks}
    for level_idx, level in enumerate(levels):
        console.print(f"  [dim]level {level_idx}[/dim]")
        for task_id in level:
            task = task_by_id[task_id]
            deps = ", ".join(task.dependencies) if task.dependencies else "—"
            console.print(
                f"    [cyan]{task.id}[/cyan]  {task.type}  "
                f"[dim]deps=[{deps}][/dim]"
            )
            if task.description:
                console.print(f"      {task.description}")

    if plan.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in plan.notes:
            console.print(f"  [yellow]•[/yellow] {note}")


@plan_app.command("create")
def plan_create(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a labpilot.yaml workspace)",
    ),
    hypothesis_id: str | None = typer.Option(
        None, "--hypothesis", "-H", help="Hypothesis ID (e.g. H-001)"
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Create P-001 baseline plan from Analyze (no hypothesis)",
    ),
    priority: int = typer.Option(0, "--priority", help="Plan priority (higher = sooner)"),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print: text, json, or markdown",
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
) -> None:
    """Compile a ResearchPlan DAG (plan-only; no execution).

    Use ``--hypothesis H-xxx`` for improvement plans, or ``--baseline`` for the
    first P-001 Analyze-derived baseline. Mutually exclusive.
    """
    output_format = _validate_format(output_format)
    if baseline and hypothesis_id:
        console.print(
            "[red]Use either --baseline or --hypothesis, not both.[/red]"
        )
        raise typer.Exit(code=1)
    if not baseline and not hypothesis_id:
        console.print(
            "[red]Provide --baseline or --hypothesis H-xxx.[/red]"
        )
        raise typer.Exit(code=1)

    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    llm = resolve_llm_client(config.llm)

    try:
        if baseline:
            plan = compile_baseline_plan(
                competition,
                knowledge_dir=config.knowledge_dir,
                llm_client=llm,
                priority=priority,
            )
        else:
            assert hypothesis_id is not None
            hyp_store = HypothesisStore(config.knowledge_dir, competition)
            hypothesis = hyp_store.get(hypothesis_id)
            if hypothesis is None:
                console.print(
                    f"[red]Hypothesis not found:[/red] {hypothesis_id} "
                    f"(competition={competition})."
                )
                raise typer.Exit(code=1)
            plan = compile_research_plan(
                hypothesis,
                knowledge_dir=config.knowledge_dir,
                competition=competition,
                llm_client=llm,
                priority=priority,
            )
    except BaselinePlanError as exc:
        console.print(f"[red]Baseline plan refused:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    paths = ResearchPaths(config.knowledge_dir, competition)
    if output_format == "text":
        kind = plan.metadata.get("plan_kind", "hypothesis")
        console.print(
            f"[green]Created[/green] plan [cyan]{plan.id}[/cyan] "
            f"({kind}, {plan.generated_by}, {len(plan.tasks)} tasks)"
        )
        console.print(
            f"  projections: {paths.plans_dir / f'{plan.id}.json'} , "
            f"{paths.plans_dir / f'{plan.id}.md'}"
        )
    _print_plan(plan, output_format)


@plan_app.command("show")
def plan_show(
    plan_id: str = typer.Argument(..., help="Plan ID (e.g. P-001)"),
    competition: str | None = typer.Option(
        None,
        "--competition",
        "-c",
        help="Competition slug (optional inside a workspace)",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="What to print: text, json, or markdown",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None, "--workspace", help="Competition workspace root"
    ),
) -> None:
    """Show one research plan and its task DAG."""
    output_format = _validate_format(output_format)
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    store = PlanStore(config.knowledge_dir, competition)
    try:
        plan = store.get_plan(plan_id)
    finally:
        store.close()

    if plan is None:
        console.print(
            f"[red]Plan not found:[/red] {plan_id} (competition={competition})."
        )
        raise typer.Exit(code=1)

    _print_plan(plan, output_format)


@plan_app.command("list")
def plan_list(
    competition: str | None = typer.Argument(
        None,
        help="Competition slug (optional inside a workspace)",
    ),
    status: str | None = typer.Option(
        None,
        "--status",
        help="Filter by status: draft, ready, in_progress, done, abandoned",
    ),
    config_path: Path = typer.Option(
        Path("configs/default.yaml"), "--config", help="Path to config file"
    ),
    knowledge_dir: Path | None = typer.Option(
        None, "--knowledge-dir", help="Override knowledge directory"
    ),
    workspace_path: Path | None = typer.Option(
        None, "--workspace", help="Competition workspace root"
    ),
) -> None:
    """List research plans for a competition."""
    status_filter = _parse_status(status)
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    store = PlanStore(config.knowledge_dir, competition)
    try:
        plans = store.list_plans(status=status_filter)
    finally:
        store.close()

    if not plans:
        console.print(f"No research plans for [cyan]{competition}[/cyan].")
        raise typer.Exit()

    table = Table(title=f"Research plans: {competition}")
    table.add_column("ID", style="cyan")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Hypothesis")
    table.add_column("By")
    table.add_column("Tasks")
    table.add_column("Goal")
    for plan in plans:
        kind = str(plan.metadata.get("plan_kind") or ("hypothesis" if plan.hypothesis_id else "—"))
        table.add_row(
            plan.id,
            kind,
            str(plan.status),
            plan.hypothesis_id or "—",
            plan.generated_by,
            str(len(plan.tasks)),
            (plan.goal or "—")[:40],
        )
    console.print(table)
