"""CLI for Research Reflection — reflect / journal / claims."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from labpilot.config import load_config
from labpilot.llm.client import resolve_llm_client
from labpilot.research_engine.reflection.journal import JournalProjector
from labpilot.research_engine.reflection.pipeline import run_reflection
from labpilot.research_engine.reflection.store import ReflectionStore

reflect_app = typer.Typer(
    help="Reflect on executions and inspect the research journal.",
    no_args_is_help=True,
)
claims_app = typer.Typer(help="Inspect research claims.", no_args_is_help=True)
console = Console()


def _config(config_path: Path, knowledge_dir: Path | None):
    config = load_config(config_path)
    if knowledge_dir:
        config.knowledge_dir = knowledge_dir
    return config


@reflect_app.command("run")
def reflect_run(
    competition: str = typer.Option(..., "--competition", "-c"),
    execution: str | None = typer.Option(None, "--execution", "-e"),
    workspace: Path | None = typer.Option(None, "--workspace"),
    offline: bool = typer.Option(False, "--offline", help="Force rule_engine critic"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute without DB writes"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    """Run Evidence → Critic → Belief / Hypothesis updates for an execution."""
    if not execution and not workspace:
        raise typer.BadParameter("Provide --execution and/or --workspace")
    config = _config(config_path, knowledge_dir)
    llm = None if offline else resolve_llm_client(config.llm)
    result = run_reflection(
        config.knowledge_dir,
        competition,
        execution_id=execution,
        workspace_path=workspace,
        llm_client=llm,
        persist=not dry_run,
    )
    evidence = result.get("evidence") or {}
    console.print(
        f"[green]Reflection complete[/green] evidence={evidence.get('id')} "
        f"strength={evidence.get('strength')} "
        f"belief={ (result.get('belief') or {}).get('belief_id') }"
    )


@reflect_app.command("journal")
def journal_cmd(
    competition: str = typer.Option(..., "--competition", "-c"),
    output_json: bool = typer.Option(False, "--json"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    """Print the research journal projection for a competition."""
    config = _config(config_path, knowledge_dir)
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
    competition: str = typer.Option(..., "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    config = _config(config_path, knowledge_dir)
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
    competition: str = typer.Option(..., "--competition", "-c"),
    config_path: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    knowledge_dir: Path | None = typer.Option(None, "--knowledge-dir"),
) -> None:
    config = _config(config_path, knowledge_dir)
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
