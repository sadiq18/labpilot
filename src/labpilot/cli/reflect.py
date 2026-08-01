"""CLI for Research Reflection — reflect / journal / claims.

``reflect run`` invokes the ``reflect`` tool (Strangler Phase A).
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
from labpilot.research_engine.reflection.journal import JournalProjector
from labpilot.research_engine.reflection.store import ReflectionStore

reflect_app = typer.Typer(
    help="Reflect on executions and inspect the research journal.",
    no_args_is_help=True,
)
claims_app = typer.Typer(help="Inspect research claims.", no_args_is_help=True)
console = Console()


@reflect_app.command("run")
def reflect_run(
    competition: str | None = typer.Option(None, "--competition", "-c"),
    execution: str | None = typer.Option(None, "--execution", "-e"),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Engineer execution workspace path (metrics/artifacts)",
    ),
    offline: bool = typer.Option(False, "--offline", help="Force rule_engine critic"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute without DB writes"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    """Run Evidence → Critic → Belief / Hypothesis updates for an execution."""
    if not execution and not workspace:
        raise typer.BadParameter("Provide --execution and/or --workspace")
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, client)
    ws = resolve_os_workspace(competition=competition, config=config, client=client)
    llm = None if offline else resolve_llm_client(config.llm)
    result = default_tools().invoke(
        "reflect",
        ws,
        execution_id=execution,
        workspace_path=str(workspace) if workspace is not None else None,
        llm_client=llm,
        persist=not dry_run,
    )
    console.print(
        f"[green]Reflection complete[/green] evidence={result.data.get('evidence_id')} "
        f"strength={result.data.get('evidence_strength')} "
        f"belief={result.data.get('belief_id')}"
    )


@reflect_app.command("journal")
def journal_cmd(
    competition: str | None = typer.Option(None, "--competition", "-c"),
    output_json: bool = typer.Option(False, "--json"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    """Print the research journal projection for a competition."""
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, client)
    projector = JournalProjector(config.knowledge_dir, competition)
    try:
        if output_json:
            print(projector.render_json())
        else:
            print(projector.render_markdown())
    finally:
        projector.close()


@claims_app.command("list")
def claims_list(
    competition: str | None = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, client)
    store = ReflectionStore(config.knowledge_dir, competition)
    try:
        claims = store.list_claims()
        if not claims:
            console.print("No claims.")
            return
        for claim in claims:
            console.print(
                f"{claim['id']} [{claim['status']}] "
                f"conf={claim['confidence']:.2f} — {claim['statement']}"
            )
    finally:
        store.close()


@claims_app.command("show")
def claims_show(
    claim_id: str = typer.Argument(...),
    competition: str | None = typer.Option(None, "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    config, client = load_cli_config(
        config_path=config_path,
        knowledge_dir=knowledge_dir,
    )
    competition = resolve_competition(competition, client)
    store = ReflectionStore(config.knowledge_dir, competition)
    try:
        claim = store.get_claim(claim_id)
        if claim is None:
            raise typer.BadParameter(f"unknown claim: {claim_id}")
        print(json_dumps(claim))
        edges = store.list_claim_evidence(claim_id)
        for edge in edges:
            console.print(
                f"  {edge['relation']} {edge['evidence_id']} (w={edge['weight']})"
            )
    finally:
        store.close()


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)
