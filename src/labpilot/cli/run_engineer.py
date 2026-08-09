"""Plan-driven ``research run`` / ``research resume`` (Research Engineer).

``research run`` invokes the ``run_plan`` tool (Strangler Phase A).
``research resume`` still uses the Engineer directly (no resume tool yet).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.cli.config_helpers import (
    default_tools,
    load_cli_config,
    resolve_competition,
    resolve_os_workspace,
)
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.artifacts.execution import ExecutionArtifacts
from labpilot.research_engine.execution import (
    EngineerError,
    ResearchEngineer,
    default_capability_registry,
)

console = Console()


def _codegen_strategy(workspace) -> str:
    """`codegen.strategy` for this workspace — the same reader `run_plan` uses."""
    from labpilot.research_engine.tools.handlers.run import _codegen_strategy as read

    return read(workspace)


def _engineer_constraints(
    *,
    config,
    workspace,
    dry_run: bool,
    submit: bool,
    llm_client=None,
) -> dict:
    constraints = {
        # `research resume` runs the same capability as `research plan run`, so
        # it reads the same setting through the same helper. It did not, and
        # the capability's own fallback still said `whole_file`, so resume took
        # the whole-file path however the workspace was configured. Reported on
        # PR #118.
        "codegen_strategy": _codegen_strategy(workspace) if workspace is not None else "",
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
    """Execute an approved ResearchPlan via the ``run_plan`` tool."""
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, client)
    ws = resolve_os_workspace(competition=competition, config=config, client=client)
    llm = resolve_llm_client(config.llm)
    constraints = {
        "kaggle": config.kaggle,
        "profiler": config.profiler,
    }
    configs_dir = ws.root / "configs"
    if configs_dir.is_dir():
        constraints["competitions_dir"] = configs_dir

    console.print(
        f"[bold]Research Engineer[/bold] — running plan "
        f"[cyan]{plan_id}[/cyan] for [cyan]{competition}[/cyan]"
        + (" [yellow](dry-run)[/yellow]" if dry_run else "")
    )
    try:
        result = default_tools().invoke(
            "run_plan",
            ws,
            plan_id=plan_id,
            dry_run=dry_run,
            submit=submit,
            install_packages=install_packages,
            llm_client=llm,
            constraints=constraints,
        )
    except EngineerError as exc:
        message = str(exc)
        if message.lower().startswith("plan not found"):
            console.print(
                f"[red]Plan not found:[/red] {plan_id} (competition={competition}). "
                "Create one with `research plan create --baseline` "
                "or `--hypothesis H-xxx`."
            )
        else:
            console.print(f"[red]Engineer error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    status = result.data.get("status")
    execution_id = result.data.get("execution_id")
    color = "green" if status == "succeeded" else "red"
    console.print(f"\n[{color}]Execution {execution_id}:[/{color}] {status}")
    if result.data.get("workspace_path"):
        console.print(f"  workspace: {result.data['workspace_path']}")
    if result.data.get("error"):
        console.print(f"  error: {result.data['error']}")


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
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, client)
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
            workspace=client,
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
