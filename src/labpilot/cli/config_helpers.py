"""Shared CLI config + competition resolution with workspace discovery."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.config import AppConfig
from labpilot.research_engine.tools import ToolRegistry, build_default_tool_registry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import (
    CompetitionWorkspace,
    load_config_for_cwd,
    resolve_competition_arg,
)

console = Console()


def load_cli_config(
    *,
    config_path: Path | None = None,
    knowledge_dir: Path | None = None,
    runs_dir: Path | None = None,
    workspace_path: Path | None = None,
) -> tuple[AppConfig, CompetitionWorkspace | None]:
    """Load AppConfig applying ``labpilot.yaml`` when present under CWD."""
    try:
        return load_config_for_cwd(
            config_path=config_path,
            knowledge_dir=knowledge_dir,
            runs_dir=runs_dir,
            workspace_path=workspace_path,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Config/workspace error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def resolve_competition(
    competition: str | None,
    workspace: CompetitionWorkspace | None,
    *,
    required: bool = True,
) -> str:
    """Default slug from workspace; exit with a clear message on mismatch/missing."""
    try:
        return resolve_competition_arg(competition, workspace, required=required)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


def resolve_os_workspace(
    *,
    competition: str,
    config: AppConfig,
    client: CompetitionWorkspace | None,
    runs_dir: Path | None = None,
    goal: str | None = None,
) -> Workspace:
    """Build the Research OS :class:`Workspace` facade for tool invocations."""
    effective_runs = runs_dir if runs_dir is not None else config.runs_dir
    if client is not None:
        return Workspace.from_client(client, goal=goal, runs_dir=effective_runs)
    return Workspace.from_competition(
        config.knowledge_dir,
        competition,
        goal=goal,
        runs_dir=effective_runs,
    )


def default_tools() -> ToolRegistry:
    """Return the default in-process tool registry (CLI strangler entry point)."""
    return build_default_tool_registry()
