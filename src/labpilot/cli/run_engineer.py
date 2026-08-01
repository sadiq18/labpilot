"""Plan-driven ``research run`` / ``research resume`` (Research Engineer)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import load_cli_config, resolve_competition
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.artifacts.execution import ExecutionArtifacts
from labpilot.research_engine.artifacts.plan import PlanArtifacts
from labpilot.research_engine.execution import (
    EngineerError,
    ResearchEngineer,
    default_capability_registry,
)

console = Console()


def _engineer_constraints(
    *,
    config,
    workspace,
    dry_run: bool,
    submit: bool,
    llm_client=None,
) -> dict:
    constraints = {
        "dry_run": dry_run,
        "allow_upload": submit,
        "smoke_syntax_only": dry_run,
        "train_stub": dry_run,
        "remote_dry_run": True,
        "skip_download": dry_run,
        "kaggle": config.kaggle,
        "profiler": config.profiler,
        "llm_client": llm_client,
    }
    if workspace is not None:
        constraints["competitions_dir"] = workspace.root / "configs"
    return constraints


def run_plan_command(
    *,
    plan_id: str,
    competition: str | None,
    config_path: Path,
    knowledge_dir: Path | None,
    dry_run: bool,
    submit: bool,
    install_packages: bool,
    workspace_path: Path | None = None,
) -> None:
    """Execute an approved ResearchPlan via the Research Engineer."""
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    plan_arts = PlanArtifacts(config.knowledge_dir, competition)
    try:
        plan = plan_arts.get(plan_id)
    finally:
        plan_arts.close()

    if plan is None:
        console.print(
            f"[red]Plan not found:[/red] {plan_id} (competition={competition}). "
            "Create one with `research plan create --baseline` "
            "or `--hypothesis H-xxx`."
        )
        raise typer.Exit(code=1)

    llm = resolve_llm_client(config.llm)
    registry = default_capability_registry(
        install_packages=install_packages and not dry_run,
        llm_client=llm,
    )
    exec_arts = ExecutionArtifacts(config.knowledge_dir, competition)
    engineer = ResearchEngineer(
        knowledge_dir=config.knowledge_dir,
        competition=competition,
        registry=registry,
        execution_store=exec_arts.store,
        constraints=_engineer_constraints(
            config=config,
            workspace=workspace,
            dry_run=dry_run,
            submit=submit,
            llm_client=llm,
        ),
    )
    try:
        console.print(
            f"[bold]Research Engineer[/bold] — running plan "
            f"[cyan]{plan_id}[/cyan] for [cyan]{competition}[/cyan]"
            + (" [yellow](dry-run)[/yellow]" if dry_run else "")
        )
        execution, _ref = exec_arts.create(plan_id)
        execution = engineer.run_plan(plan_id, execution=execution)
    except EngineerError as exc:
        console.print(f"[red]Engineer error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        engineer.close()
        exec_arts.close()

    color = "green" if execution.status == "succeeded" else "red"
    console.print(
        f"\n[{color}]Execution {execution.id}:[/{color}] {execution.status}"
    )
    if execution.workspace_path:
        console.print(f"  workspace: {execution.workspace_path}")
    if execution.error:
        console.print(f"  error: {execution.error}")


def resume_execution_command(
    *,
    execution_id: str,
    competition: str | None,
    config_path: Path,
    knowledge_dir: Path | None,
    dry_run: bool,
    submit: bool,
    install_packages: bool,
    workspace_path: Path | None = None,
) -> None:
    config, workspace = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, workspace)
    exec_arts = ExecutionArtifacts(config.knowledge_dir, competition)
    try:
        existing = exec_arts.get(execution_id)
    finally:
        exec_arts.close()
    if existing is None:
        console.print(
            f"[red]Execution not found:[/red] {execution_id} "
            f"(competition={competition})."
        )
        raise typer.Exit(code=1)

    llm = resolve_llm_client(config.llm)
    registry = default_capability_registry(
        install_packages=install_packages and not dry_run,
        llm_client=llm,
    )
    engineer = ResearchEngineer(
        knowledge_dir=config.knowledge_dir,
        competition=competition,
        registry=registry,
        constraints=_engineer_constraints(
            config=config,
            workspace=workspace,
            dry_run=dry_run,
            submit=submit,
            llm_client=llm,
        ),
    )
    try:
        console.print(
            f"[bold]Research Engineer[/bold] — resuming "
            f"[cyan]{execution_id}[/cyan]"
        )
        execution = engineer.resume(execution_id)
    except EngineerError as exc:
        console.print(f"[red]Engineer error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        engineer.close()

    color = "green" if execution.status == "succeeded" else "red"
    console.print(
        f"\n[{color}]Execution {execution.id}:[/{color}] {execution.status}"
    )
