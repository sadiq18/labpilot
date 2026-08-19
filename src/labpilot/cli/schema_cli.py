"""``research schema`` — what the profiler concluded, and what it could not.

The operator's side of M22: one command to see every answer with the evidence
behind it, and one to settle a question the profiler is not entitled to answer
for you.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from labpilot.accessor.profiler.questions import (
    BLOCKING_FIELDS,
    load_answers,
    pending_schema_questions,
    record_answer,
)
from labpilot.accessor.profiler.tabular import REQUIRED_FIELDS, DatasetProfile

schema_app = typer.Typer(
    help="Inspect the dataset schema, and answer what the profiler could not.",
    no_args_is_help=True,
)
console = Console()


def _load(workspace: Path) -> DatasetProfile:
    path = workspace / "profile.json"
    if not path.is_file():
        console.print(f"[red]No profile.json under {workspace}[/red]")
        console.print("Run a campaign, or `research conduct`, to build one.")
        raise typer.Exit(code=1)
    try:
        return DatasetProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        console.print(f"[red]{path} is not a profile this version can read: {exc}[/red]")
        raise typer.Exit(code=1) from exc


def _value_of(profile: DatasetProfile, field: str) -> str:
    if field == "id_columns":
        return ", ".join(profile.id_columns) or "—"
    value = getattr(profile, field, None)
    if field == "metric" and profile.metric is not None:
        return f"{profile.metric.name} ({profile.metric.direction or 'direction unknown'})"
    return str(value) if value not in (None, "") else "—"


@schema_app.command("show")
def show(
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Competition workspace"),
) -> None:
    """Every answer, how sure it is, and what it was concluded from."""
    profile = _load(workspace)
    answers = load_answers(workspace)

    table = Table(title=f"Dataset schema — {profile.competition}", show_lines=False)
    table.add_column("Field", overflow="fold")
    table.add_column("Value", overflow="fold")
    table.add_column("Conf", justify="right")
    table.add_column("Band")
    table.add_column("Evidence", overflow="fold")
    for field in REQUIRED_FIELDS:
        inference = profile.inferences.get(field)
        evidence = (
            ", ".join(signal.id for signal in inference.signals) if inference else "none recorded"
        )
        table.add_row(
            field,
            _value_of(profile, field),
            f"{profile.confidence_in(field):.2f}",
            inference.band if inference else "—",
            evidence or "nothing fired",
        )
    console.print(table)
    console.print(f"Weakest answer: [bold]{profile.confidence:.2f}[/bold]")

    questions = pending_schema_questions(profile, answers)
    if not questions:
        console.print("[green]No open questions.[/green]")
        return
    for question in questions:
        console.print(f"\n[yellow]Open question:[/yellow] {question.field} — {question.context}")
        console.print(f"  provisional: {question.provisional or '—'} (not acted on)")
        for candidate in question.candidates:
            fired = ", ".join(signal.id for signal in candidate.signals) or "nothing fired"
            console.print(f"  · {candidate.candidate} ({candidate.confidence:.2f}) — {fired}")
        console.print(f"  answer with: research schema answer {question.field} <value>")


@schema_app.command("answer")
def answer(
    field: str = typer.Argument(..., help=f"One of: {', '.join(BLOCKING_FIELDS)}"),
    value: str = typer.Argument(..., help="The column name that is correct"),
    workspace: Path = typer.Option(Path("."), "--workspace", "-w", help="Competition workspace"),
) -> None:
    """Settle a question. The answer outlives the profile it was asked about."""
    try:
        answers = record_answer(workspace, field, value)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Recorded[/green] {field} = {value!r}")
    console.print(f"Answers on file: {json.dumps(answers, sort_keys=True)}")
    console.print("The next profile rebuild will use it; run `research schema show` to confirm.")
