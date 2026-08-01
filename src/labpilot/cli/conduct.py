"""``research conduct`` — product entry for the Research Conductor."""

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
from labpilot.research_engine.conductor import ConductorStore, run_until_stop
from labpilot.research_engine.conductor.approvals import ApprovalResult
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.workspace_facade import Workspace

conduct_app = typer.Typer(
    help="Run the Research Conductor goal loop (product entry).",
    no_args_is_help=True,
)
console = Console()


def _test_registry_subset() -> object:
    """Build a registry with cheap offline-safe tools for dry/offline loops."""
    from labpilot.research_engine.tools.handlers.papers import search_papers
    from labpilot.research_engine.tools.handlers.memory import query_memory
    from labpilot.research_engine.tools.registry import ToolRegistry

    reg = ToolRegistry()

    def echo_analyze(workspace: Workspace, **_kwargs: object) -> ToolResult:
        path = workspace.research_paths.reports_dir
        path.mkdir(parents=True, exist_ok=True)
        out = path / "analyze.json"
        if not out.is_file():
            out.write_text(
                '{"schema_version":1,"competition":{"slug":"%s"},"analyzers":[],"notes":["conduct stub"]}\n'
                % workspace.competition,
                encoding="utf-8",
            )
        return ToolResult(
            refs=[],
            data={"path": str(out), "stub": True},
        )

    reg.register(
        ToolDescriptor(
            name="analyze_competition",
            description="stub",
            handler=echo_analyze,
        )
    )
    reg.register(
        ToolDescriptor(
            name="search_papers",
            description="search papers",
            handler=lambda ws, **kw: search_papers(ws, offline=True, **kw),
        )
    )
    reg.register(
        ToolDescriptor(
            name="query_memory",
            description="query memory",
            handler=query_memory,
        )
    )
    return reg


@conduct_app.command("run")
def conduct_run(
    goal: str = typer.Argument(..., help='Research goal, e.g. "Win Rogii"'),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    max_steps: int = typer.Option(8, "--max-steps", help="Stop after N policy steps"),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-approve gated tools (no interactive prompts)",
    ),
    offline: bool = typer.Option(
        False,
        "--offline",
        help="No LLM policy; deterministic catalog order + safe stubs where needed",
    ),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
    workspace_path: Path | None = typer.Option(None, "--workspace"),
) -> None:
    """Observe → think → enqueue → approve → execute until stop or max steps."""
    config, marker_ws = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    competition = resolve_competition(competition, marker_ws)
    ws = resolve_os_workspace(
        competition=competition,
        config=config,
        client=marker_ws,
        goal=goal,
    )
    ws.ensure_roots()

    store = ConductorStore(ws.knowledge_dir, competition)
    registry = default_tools()
    llm = None if offline else resolve_llm_client(config.llm)
    if offline:
        # Prefer cheap tools so CI / dry runs don't need full analyze/plan/run.
        registry = _test_registry_subset()  # type: ignore[assignment]

    def _approve(tool_name: str) -> ApprovalResult:
        if yes:
            return ApprovalResult(decision="approve", comment="", gated_tool=tool_name)
        console.print(f"\n[yellow]Approval required[/yellow] for [cyan]{tool_name}[/cyan]")
        raw = typer.prompt("Approve? [y/N]", default="n")
        approved = str(raw).strip().lower() in {"y", "yes"}
        comment = typer.prompt(
            "Comment (optional, feeds future decisions)", default="", show_default=False
        )
        return ApprovalResult(
            decision="approve" if approved else "reject",
            comment=str(comment or ""),
            gated_tool=tool_name,
        )

    try:
        session = store.create_session(
            goal,
            metadata={"max_steps": max_steps, "offline": offline},
        )
        console.print(
            f"[bold]Conductor[/bold] session [cyan]{session.id}[/cyan] "
            f"goal={goal!r} competition={competition}"
        )
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            registry,
            llm_client=llm,
            max_steps=max_steps,
            auto_approve=yes,
            approval_prompt=None if yes else _approve,
            on_progress=lambda msg: console.print(f"  {msg}"),
        )
    finally:
        store.close()

    console.print(f"\n[green]Done[/green] — {len(decisions)} decision(s) recorded")
    for d in decisions:
        stop = " stop" if d.stop else ""
        tool = d.tool_name or "—"
        console.print(f"  {d.id}: {tool}{stop} — {d.rationale[:80]}")
