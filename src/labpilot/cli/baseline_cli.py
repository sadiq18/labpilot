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
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from labpilot.research_engine.execution.baseline.baseline_one import (
    ModelReading,
    load_baseline_one,
)
from labpilot.research_engine.execution.baseline.floor import FloorReading, load_floor
from labpilot.research_engine.execution.baseline.gate import (
    GateVerdict,
    Waiver,
    enforcement_enabled,
    evaluate_gate,
    reading_fingerprint,
    write_waiver,
)
from labpilot.research_engine.execution.baseline.report import build_report
from labpilot.workspace import discover_workspace

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
    # Two situations reach `awaiting_ml` — a model that ran and could not, and
    # no model reading at all — and they have opposite answers. `_next_step`
    # tells them apart from the readings it already has; this is the fallback
    # for when it cannot.
    "awaiting_ml": "nothing has been compared to the floor",
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
    """The workspace to report on, or None when there is not one.

    A path handed in gets the same existence check as one discovered. Without
    it, `--workspace /tmp/typo` read as a workspace whose campaign had not run
    yet — every reader below treats absent files as absent *state* — so the
    command exited 0 and sent the operator to `research conduct`, which would
    not have helped.

    `discover_workspace` is imported at module scope on purpose. A
    function-local import rebinds the real one on every call, so a test patching
    this module's attribute patched something nothing read — and passed only
    because discovery happened to find nothing from the directory the suite ran
    in.
    """
    if workspace is not None:
        return workspace if workspace.is_dir() else None
    found = discover_workspace()
    if found is None or not found.root:
        return None
    root = Path(found.root)
    return root if root.is_dir() else None


def _next_step(verdict: GateVerdict, floor: FloorReading | None, model: ModelReading | None) -> str:
    """What to do, decided from the readings rather than the state alone.

    `_NEXT_STEP` keys on the state, and two of the nine cover situations with
    **opposite** answers. `awaiting_ml` is both "a generic model cannot run on
    this dataset" — where there is nothing to do — and "no Baseline 1 has been
    taken yet", where running one is exactly the thing to do. A lookup on the
    state said the first for both, contradicting the verdict's own reason two
    lines above it in the same output.

    `show` holds the readings, so it can tell them apart instead of hedging.
    """
    if verdict.state == "awaiting_ml":
        if model is None:
            return "run the baseline — no generic model has been measured here yet"
        return f"nothing to do — {model.undefined_reason or 'a generic model cannot run here'}"
    if verdict.state == "floor_undefined" and floor is not None and floor.undefined_reason:
        return f"nothing to do — {floor.undefined_reason}"
    return _NEXT_STEP.get(verdict.state, "unknown state")


def _refuse(workspace: Path | None) -> NoReturn:
    """Say which of the two situations this is, and stop.

    "No workspace found" is the wrong sentence for a path the operator typed:
    they know where they meant, and the useful fact is that it is not there.
    """
    if workspace is not None:
        console.print(f"[red]Not a directory:[/red] {workspace}")
    else:
        console.print("[red]No workspace found.[/red] Run from inside one, or pass --workspace.")
    raise typer.Exit(2)


@baseline_app.command("show")
def show(
    workspace: Path = typer.Option(
        None, "--workspace", "-w", help="Workspace root (default: discover from cwd)"
    ),
) -> None:
    """Every strategy tried, the model's number, the verdict, and what to do."""
    root = _root(workspace)
    if root is None:
        _refuse(workspace)

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

    # The verdict's own comparison, not a second one computed from the same
    # inputs. Recomputing it also made the *suppression* implicit: `stale` and
    # `awaiting_ml` never reach the compare step, so `direction` is empty, and
    # that empty string was the only thing stopping a comparison those states
    # must not show.
    if verdict.comparison.metric_name and not verdict.comparison.incomparable_reason:
        console.print("\n" + verdict.comparison.render())

    if verdict.state == "failed":
        console.print("\n" + build_report(root, verdict, competition=root.name).render())

    console.print(f"\n[bold]Next:[/bold] {_next_step(verdict, floor, model)}")
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
        _refuse(workspace)

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
