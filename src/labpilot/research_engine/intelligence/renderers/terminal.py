"""Terminal renderer — human-facing view of an ``AnalysisReport``.

Plan 1 envelope + counts; Plan 10 adds a short top-N hypothesis preview.
Mockup-parity polish remains Plan 11.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from labpilot.research_engine.intelligence.models import AnalysisReport


def render_terminal(report: AnalysisReport, *, console: Console | None = None) -> None:
    console = console or Console()
    slug = report.competition.get("slug", "?")
    console.print(f"\n[bold]Research analysis[/bold] — [cyan]{slug}[/cyan]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Section", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Analyzers run", str(len(report.analyzers)))
    table.add_row("Artifacts", str(len(report.artifacts)))
    table.add_row("Papers", str(len(report.papers)))
    table.add_row("Repositories", str(len(report.repositories)))
    table.add_row("Related competitions", str(len(report.related_competitions)))
    table.add_row("Knowledge units", str(len(report.knowledge_units)))
    table.add_row("Hypothesis recommendations", str(len(report.hypothesis_recommendations)))
    console.print(table)

    if report.analyzers:
        console.print(f"\n[dim]Analyzers:[/dim] {', '.join(report.analyzers)}")

    if report.hypothesis_recommendations:
        console.print("\n[bold]Suggested Next Experiments[/bold]")
        for card in report.hypothesis_recommendations[:10]:
            rank = card.get("rank", "?")
            title = card.get("title") or card.get("prediction") or "(untitled)"
            impact = card.get("expected_impact", "unknown")
            confidence = card.get("confidence", 0.0)
            effort = card.get("implementation_effort", "unknown")
            hyp_id = card.get("hypothesis_id") or ""
            console.print(
                f"  [cyan]#{rank}[/cyan] {title}"
                + (f"  [dim]({hyp_id})[/dim]" if hyp_id else "")
            )
            console.print(
                f"      impact={impact}  confidence={confidence:.2f}  effort={effort}"
            )

    if report.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in report.notes:
            console.print(f"  [yellow]•[/yellow] {note}")
