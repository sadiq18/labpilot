"""``research baseline`` — what the floor scored, and whether the model beat it.

M23 §9. Until now the gate reached a verdict that nobody could see: `build_report`
had exactly one importer and it was a test file, so a campaign that failed
produced a judgement with no surface. This is that surface.

`show` is deliberately read-only and computes nothing. It reports what is on
disk, and says plainly when there is nothing there — a command that quietly
fitted five models to answer "what happened?" would be a different command.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.research_engine.execution.baseline.baseline_one import compare, load_baseline_one
from labpilot.research_engine.execution.baseline.floor import load_floor
from labpilot.research_engine.execution.baseline.gate import (
    Waiver,
    enforcement_enabled,
    evaluate_gate,
    reading_fingerprint,
    write_waiver,
)
from labpilot.research_engine.execution.baseline.report import build_report

baseline_app = typer.Typer(
    help="The floor, the generic model, and the gate's verdict over them.",
    no_args_is_help=True,
)
console = Console()

#: What the operator does about each state. The whole reason there are nine
#: rather than a boolean: M20's finding is that collapsing them is how eight
#: gates reported `pass` on things that could not run, and a verdict an operator
#: cannot act on is a wall.
_NEXT_STEP = {
    "unknown": "run `research conduct` — no validation plan exists yet",
    "floor_missing": "run the baseline; nothing has measured this dataset",
    "floor_undefined": "nothing to do — this target has no defined floor",
    "blocked_uncertain": "answer the schema question with `research schema answer`",
    "awaiting_ml": "nothing to do — a generic model cannot run on this dataset",
    "stale": "re-run the baseline; the dataset or the answers have moved",
    "failed": "read the causes below before doing anything else",
    "passed": "proceed",
    "waived": "proceed — someone accepted the failure in writing",
}

_STATE_STYLE = {
    "passed": "green",
    "waived": "yellow",
    "failed": "red",
    "blocked_uncertain": "yellow",
    "stale": "yellow",
}


def _root(workspace: Path | None) -> Path | None:
    if workspace is not None:
        return workspace
    from labpilot.workspace import discover_workspace

    found = discover_workspace()
    return Path(found.root) if found is not None and found.root else None


@baseline_app.command("show")
def show(
    workspace: Path = typer.Option(
        None, "--workspace", "-w", help="Workspace root (default: discover from cwd)"
    ),
) -> None:
    """Every strategy tried, the model's number, the verdict, and what to do."""
    root = _root(workspace)
    if root is None:
        console.print("[red]No workspace found.[/red] Run from inside one, or pass --workspace.")
        raise typer.Exit(2)

    floor, model = load_floor(root), load_baseline_one(root)
    verdict = evaluate_gate(root, enforced=enforcement_enabled())

    style = _STATE_STYLE.get(verdict.state, "cyan")
    console.print(f"\n[bold]{root.name}[/bold] — [{style}]{verdict.state}[/{style}]")
    console.print(f"  [dim]{verdict.reason}[/dim]")

    if floor is not None and floor.strategies:
        table = Table("strategy", "score", title="Floor — every constant tried", title_style="")
        for name, score in sorted(floor.strategies.items(), key=lambda kv: kv[0]):
            won = name == floor.best_strategy
            table.add_row(
                f"[green]{name}[/green]" if won else name,
                f"[green]{score:.4f}[/green]" if won else f"{score:.4f}",
            )
        console.print(table)
    elif floor is not None and floor.undefined_reason:
        console.print(f"  [dim]no floor:[/dim] {floor.undefined_reason}")

    if model is not None and model.is_defined:
        console.print(f"  [dim]generic model:[/dim] {model.model} {model.score:.4f}")
    elif model is not None and model.undefined_reason:
        console.print(f"  [dim]no generic model:[/dim] {model.undefined_reason}")

    if floor is not None and model is not None:
        comparison = compare(floor, model, verdict.comparison.direction or "")
        if not comparison.incomparable_reason:
            console.print("\n" + comparison.render())

    if verdict.state == "failed":
        console.print("\n" + build_report(root, verdict, competition=root.name).render())

    console.print(f"\n[bold]Next:[/bold] {_NEXT_STEP.get(verdict.state, 'unknown state')}")
    if not enforcement_enabled() and verdict.blocks_research:
        # Observe-only has to say so, or an operator reads a red verdict as a
        # campaign that has been stopped and goes looking for what blocked it.
        console.print(
            "[dim]The gate is observing only: this verdict is recorded and "
            "nothing is being withheld.[/dim]"
        )


@baseline_app.command("waive")
def waive(
    reason: str = typer.Argument(..., help="Why this failure is being accepted"),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Workspace root"),
    by: str = typer.Option("", "--by", help="Who accepted it"),
) -> None:
    """Accept a failing gate, in writing, against this dataset only.

    Fingerprinted on purpose. A waiver that outlived its cause would be the gate
    quietly switching itself off, which is what an env-var kill switch does — it
    gets set once during a frustrating afternoon and never unset, and nothing
    records that it happened.
    """
    from datetime import UTC, datetime

    root = _root(workspace)
    if root is None:
        console.print("[red]No workspace found.[/red] Run from inside one, or pass --workspace.")
        raise typer.Exit(2)

    verdict = evaluate_gate(root)
    if verdict.state not in ("failed", "waived"):
        console.print(
            f"[yellow]Nothing to waive:[/yellow] the gate is {verdict.state}, not failed."
        )
        raise typer.Exit(1)

    path = write_waiver(
        root,
        Waiver(
            reason=reason,
            granted_by=by,
            granted_at=datetime.now(UTC).isoformat(),
            fingerprint=reading_fingerprint(root),
        ),
    )
    console.print(f"[yellow]Waived.[/yellow] {path}")
    console.print(
        "[dim]It applies to this dataset as it stands. Re-profiling, or answering a "
        "schema question, invalidates it.[/dim]"
    )
