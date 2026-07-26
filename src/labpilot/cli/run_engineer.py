"""Plan-driven ``research run`` / ``research resume`` (Research Engineer)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.config import load_config
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.execution import (
    EngineerError,
    ResearchEngineer,
    default_capability_registry,
)
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.planner.store import PlanStore

console = Console()


def _load_config(
    config_path: Path,
    knowledge_dir: Path | None,
):
    config = load_config(config_path)
    if knowledge_dir:
        config.knowledge_dir = knowledge_dir
    return config


def run_plan_command(
    *,
    plan_id: str,
    competition: str,
    config_path: Path,
    knowledge_dir: Path | None,
    dry_run: bool,
    submit: bool,
    install_packages: bool,
) -> None:
    """Execute an approved ResearchPlan via the Research Engineer."""
    config = _load_config(config_path, knowledge_dir)
    store = PlanStore(config.knowledge_dir, competition)
    try:
        plan = store.get_plan(plan_id)
    finally:
        store.close()

    if plan is None:
        console.print(
            f"[red]Plan not found:[/red] {plan_id} (competition={competition}). "
            "Create one with `research plan create <slug> --baseline` "
            "or `--hypothesis H-xxx`."
        )
        raise typer.Exit(code=1)

    llm = resolve_llm_client(config.llm)
    registry = default_capability_registry(
        install_packages=install_packages and not dry_run,
        llm_client=llm,
    )
    constraints = {
        "dry_run": dry_run,
        "allow_upload": submit,
        "smoke_syntax_only": dry_run,
        "train_stub": dry_run,
        "remote_dry_run": True,
        "skip_download": dry_run,
        "kaggle": config.kaggle,
        "profiler": config.profiler,
    }
    engineer = ResearchEngineer(
        knowledge_dir=config.knowledge_dir,
        competition=competition,
        registry=registry,
        constraints=constraints,
    )
    try:
        console.print(
            f"[bold]Research Engineer[/bold] — running plan "
            f"[cyan]{plan_id}[/cyan] for [cyan]{competition}[/cyan]"
            + (" [yellow](dry-run)[/yellow]" if dry_run else "")
        )
        execution = engineer.run_plan(plan_id)
    except EngineerError as exc:
        console.print(f"[red]Engineer error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        engineer.close()

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
    competition: str,
    config_path: Path,
    knowledge_dir: Path | None,
    dry_run: bool,
    submit: bool,
    install_packages: bool,
) -> None:
    config = _load_config(config_path, knowledge_dir)
    exec_store = ExecutionStore(config.knowledge_dir, competition)
    try:
        existing = exec_store.get_execution(execution_id)
    finally:
        exec_store.close()
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
        constraints={
            "dry_run": dry_run,
            "allow_upload": submit,
            "smoke_syntax_only": dry_run,
            "train_stub": dry_run,
            "remote_dry_run": True,
            "skip_download": dry_run,
            "kaggle": config.kaggle,
            "profiler": config.profiler,
        },
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
