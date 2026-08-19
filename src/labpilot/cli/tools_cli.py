"""``research tools`` — list catalog, export gaps, maintainer gap decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from labpilot.cli.config_helpers import (
    default_tools,
    load_cli_config,
    resolve_competition,
    resolve_os_workspace,
)
from labpilot.research_engine.conductor.gap_ledger import (
    apply_gap_decision,
    export_gaps_payload,
    is_maintainer_enabled,
)
from labpilot.research_engine.conductor.store import ConductorStore

tools_app = typer.Typer(
    help=(
        "Tool catalog and capability gaps. "
        "Promote/defer/reject require LABPILOT_MAINTAINER=1."
    ),
    no_args_is_help=True,
)
console = Console()


def _open_store(
    *,
    competition: str | None,
    config_path: Path,
    knowledge_dir: Path | None,
    workspace_path: Path | None,
) -> tuple[ConductorStore, str]:
    config, marker_ws = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    slug = resolve_competition(competition, marker_ws)
    ws = resolve_os_workspace(
        competition=slug,
        config=config,
        client=marker_ws,
        goal="",
    )
    ws.ensure_roots()
    return ConductorStore(ws.knowledge_dir, slug), slug


def _require_maintainer() -> None:
    if not is_maintainer_enabled():
        console.print(
            "[red]Maintainer only.[/red] Set "
            "[cyan]LABPILOT_MAINTAINER=1[/cyan] for gap decisions. "
            "End users: use [cyan]research tools export-gaps[/cyan]."
        )
        raise typer.Exit(1)


#: How each status reads to an operator deciding whether to trust a tool.
_STATUS_STYLE = {
    "real": "green",
    "partial": "yellow",
    "fixed": "dim",
}


@tools_app.command("list")
def tools_list() -> None:
    """List tools in the default Conductor catalog, with capability status.

    M15 exit criterion 2: an operator can see which tools can actually change
    an outcome without reading source. `varies_by` names the inputs proven —
    by `tests/unit/test_tool_contracts.py`, not by assertion — to change what
    the tool produces.
    """
    registry = default_tools()
    table = Table(title="Registered tools")
    table.add_column("name")
    table.add_column("status")
    table.add_column("varies by")
    table.add_column("description")
    for desc in registry.list_tools():
        status = desc.capability_status
        table.add_row(
            desc.name,
            f"[{_STATUS_STYLE.get(status, 'white')}]{status}[/]",
            ", ".join(desc.varies_by) or "[dim]—[/dim]",
            (desc.description or "")[:60],
        )
    console.print(table)
    console.print(
        "[dim]real = a declared input provably changes the output · "
        "partial = degrades honestly when unavailable · "
        "fixed = same output regardless of input[/dim]"
    )


@tools_app.command("gaps")
def tools_gaps(
    status: Optional[str] = typer.Option(
        "open",
        "--status",
        help="Filter by status (open|promoted|deferred|rejected|alias); empty=all",
    ),
    promote: Optional[str] = typer.Option(
        None,
        "--promote",
        help="Maintainer: mark gap_key promoted (requires --tool)",
    ),
    alias: Optional[str] = typer.Option(
        None,
        "--alias",
        help="Maintainer: mark gap_key as alias of --tool",
    ),
    defer: Optional[str] = typer.Option(
        None,
        "--defer",
        help="Maintainer: defer gap_key",
    ),
    reject: Optional[str] = typer.Option(
        None,
        "--reject",
        help="Maintainer: reject gap_key",
    ),
    tool: Optional[str] = typer.Option(
        None,
        "--tool",
        help="Tool name for --promote / --alias",
    ),
    reason: str = typer.Option("", "--reason", help="Decision reason"),
    competition: Optional[str] = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Optional[Path] = typer.Option(None, "--knowledge-dir"),
    workspace_path: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """List local capability gaps; decisions require LABPILOT_MAINTAINER=1."""
    decision_args = [promote, alias, defer, reject]
    active = [a for a in decision_args if a]
    if len(active) > 1:
        console.print("[red]Specify only one of --promote/--alias/--defer/--reject[/red]")
        raise typer.Exit(1)

    store, slug = _open_store(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    try:
        if promote or alias or defer or reject:
            _require_maintainer()
            if promote:
                gap_key, decision = promote, "promote"
            elif alias:
                gap_key, decision = alias, "alias"
            elif defer:
                gap_key, decision = defer, "defer"
            else:
                gap_key, decision = reject, "reject"  # type: ignore[assignment]
            assert gap_key is not None
            try:
                record = apply_gap_decision(
                    store,
                    gap_key,
                    decision,  # type: ignore[arg-type]
                    reason=reason,
                    promoted_tool=tool,
                )
            except (PermissionError, KeyError, ValueError) as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc
            console.print(
                f"[green]{record.decision}[/green] {record.gap_key} "
                f"(tool={record.promoted_tool or '-'}) — still implement via PR "
                "to ship in LabPilot for all users"
            )
            return

        filter_status = status if status else None
        gaps = store.list_capability_gaps(status=filter_status)
        if not gaps:
            console.print(f"[dim]No gaps[/dim] (competition={slug}, status={status!r})")
            return
        table = Table(title=f"Capability gaps ({slug})")
        table.add_column("gap_key")
        table.add_column("count", justify="right")
        table.add_column("status")
        table.add_column("last_seen")
        table.add_column("promoted_tool")
        for g in gaps:
            table.add_row(
                g.gap_key,
                str(g.count),
                g.status,
                g.last_seen_at,
                g.promoted_tool or "",
            )
        console.print(table)
        if not is_maintainer_enabled():
            console.print(
                "[dim]Decisions: LABPILOT_MAINTAINER=1 research tools gaps "
                "--promote KEY --tool NAME[/dim]"
            )
    finally:
        store.close()


@tools_app.command("export-gaps")
def tools_export_gaps(
    output: Path = typer.Option(
        Path("capability-gaps.json"),
        "--output",
        "-o",
        help="Write redacted gap aggregate JSON",
    ),
    status: Optional[str] = typer.Option(
        "open",
        "--status",
        help="Filter by status; empty=all",
    ),
    competition: Optional[str] = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Optional[Path] = typer.Option(None, "--knowledge-dir"),
    workspace_path: Optional[Path] = typer.Option(None, "--workspace"),
) -> None:
    """Export redacted local gaps (opt-in file for maintainer / future telemetry)."""
    store, slug = _open_store(
        competition=competition,
        config_path=config_path,
        knowledge_dir=knowledge_dir,
        workspace_path=workspace_path,
    )
    try:
        filter_status = status if status else None
        payload = export_gaps_payload(
            store,
            status=filter_status,
            competition=slug,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        console.print(f"[green]Wrote[/green] {len(payload['gaps'])} gaps → [cyan]{output}[/cyan]")
    finally:
        store.close()
