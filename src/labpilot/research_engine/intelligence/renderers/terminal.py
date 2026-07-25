"""Terminal renderer — human-facing view of an ``AnalysisReport``.

Mockup-parity sections (Milestone 3 Plan 11 / README capstone vision).
JSON remains the contract; this module is presentation only.
"""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from labpilot.research_engine.intelligence.models import AnalysisReport

_FAILURE_MARKERS = (
    "hurt",
    "hurts",
    "regress",
    "failed",
    "failure",
    "worse",
    "decreased",
    "drop",
    "negative",
)


def render_terminal(report: AnalysisReport, *, console: Console | None = None) -> None:
    console = console or Console()
    slug = str(report.competition.get("slug") or "?")
    title = str(report.competition.get("title") or slug)

    console.print(f"\n[bold]Competition Summary[/bold]")
    console.print("─" * 28)
    console.print(f"  [cyan]{title}[/cyan]  ([dim]{slug}[/dim])")
    _print_profile_line(console, report)

    console.print("\n[bold]Related Competitions[/bold]")
    if report.related_competitions:
        for item in report.related_competitions[:8]:
            label = item.get("title") or item.get("slug") or "?"
            relation = item.get("relation") or ""
            suffix = f"  [dim]({relation})[/dim]" if relation else ""
            console.print(f"    {label}{suffix}")
    else:
        console.print("    [dim](none)[/dim]")

    console.print("\n[bold]Relevant Papers[/bold]")
    console.print(
        "    [dim](ranked by task · metric · domain · technique — not keywords alone)[/dim]"
    )
    if report.papers:
        for paper in report.papers[:8]:
            label = paper.get("title") or paper.get("id") or "?"
            console.print(f"    • {label}")
        if len(report.papers) > 8:
            console.print(f"    [dim]… +{len(report.papers) - 8} more[/dim]")
    else:
        console.print("    [dim](none)[/dim]")

    console.print("\n[bold]Relevant Experiments[/bold]")
    exp_ids = _experiment_labels(report)
    if exp_ids:
        console.print(f"    {', '.join(exp_ids[:8])} (local)")
    else:
        console.print("    [dim](none yet)[/dim]")

    console.print("\n[bold]Relevant Repositories[/bold]")
    console.print(f"    {len(report.repositories)}")
    if report.transfer_opportunities:
        top = report.transfer_opportunities[0]
        summary = top.get("summary") or top.get("hypothesis_hint") or "transfer opportunity"
        effort = top.get("effort") or "?"
        gain = top.get("expected_gain") or "?"
        console.print(f"    Top transfer: {summary}  [dim](~{effort}, {gain} gain)[/dim]")

    console.print("\n[bold]Relevant Discussions[/bold]")
    console.print("    [dim](when Forum Intelligence provider available)[/dim]")

    console.print("\n[bold]Relevant Failures[/bold]")
    failures = _failure_lines(report)
    if failures:
        for line in failures[:8]:
            console.print(f"    {line}")
    else:
        console.print("    [dim](none recorded)[/dim]")

    console.print("\n[bold]Winning Solutions[/bold]")
    _print_winning_solutions(console, report)

    console.print("\n[bold]Interesting Forum Discussions[/bold]")
    if report.forum_knowledge:
        for item in report.forum_knowledge[:5]:
            console.print(f"    • {item.get('title') or item.get('id') or item}")
    else:
        console.print("    [dim]Status: Unavailable[/dim]")
        console.print(
            "    [dim]Reason: Forum Intelligence — providers after spike / GitHub Issues[/dim]"
        )

    console.print("\n[bold]Known Strong Techniques[/bold]")
    console.print("    [dim](External vs Locally Validated — never auto-promote external)[/dim]")

    console.print("\n[bold]External Recommendations[/bold]")
    external = report.techniques.external_recommendations
    if external:
        # Explicit Suggested — never claim Established for external-only.
        console.print(f"    {', '.join(external[:12])}  [dim]# Suggested[/dim]")
    else:
        console.print("    [dim](none)[/dim]")

    console.print("\n[bold]Locally Validated[/bold]")
    local = report.techniques.locally_validated
    if local:
        console.print(f"    {', '.join(local[:12])}")
    else:
        console.print("    [dim](none yet — run improve to promote)[/dim]")

    if report.techniques.unverified:
        console.print("\n[bold]Unverified (local evidence / testing)[/bold]")
        console.print(f"    {', '.join(report.techniques.unverified[:12])}")

    opportunities = _opportunity_lines(report)
    console.print("\n[bold]Potential Research Opportunities[/bold]")
    if opportunities:
        for line in opportunities[:8]:
            console.print(f"    {line}")
    else:
        console.print("    [dim](see suggested experiments)[/dim]")

    console.print("\n[bold]Suggested Next Experiments[/bold]")
    console.print("    [dim]Top 10 — impact · confidence · evidence · effort[/dim]")
    console.print("    [dim](recommendations only — no autonomous planner)[/dim]")
    cards = report.hypothesis_recommendations[:10]
    if not cards:
        console.print("    [dim](none)[/dim]")
    for card in cards:
        rank = card.get("rank", "?")
        title_text = card.get("title") or card.get("prediction") or "(untitled)"
        impact = card.get("expected_impact", "unknown")
        confidence = float(card.get("confidence") or 0.0)
        effort = card.get("implementation_effort", "unknown")
        hyp_id = card.get("hypothesis_id") or ""
        evidence = card.get("supporting_evidence") or []
        console.print(
            f"  [cyan]#{rank}[/cyan] {title_text}"
            + (f"  [dim]({hyp_id})[/dim]" if hyp_id else "")
        )
        console.print(
            f"      impact={impact}  confidence={confidence:.2f}  "
            f"effort={effort}  evidence={len(evidence)}"
        )

    # Compact counts for operators who want a rollup.
    table = Table(show_header=True, header_style="bold", title="Counts")
    table.add_column("Section", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Analyzers run", str(len(report.analyzers)))
    table.add_row("Artifacts", str(len(report.artifacts)))
    table.add_row("Papers", str(len(report.papers)))
    table.add_row("Repositories", str(len(report.repositories)))
    table.add_row("Related competitions", str(len(report.related_competitions)))
    table.add_row("Knowledge units", str(len(report.knowledge_units)))
    table.add_row("Hypothesis recommendations", str(len(report.hypothesis_recommendations)))
    console.print()
    console.print(table)

    if report.analyzers:
        console.print(f"\n[dim]Analyzers:[/dim] {', '.join(report.analyzers)}")

    if report.notes:
        console.print("\n[bold]Notes[/bold]")
        for note in report.notes:
            console.print(f"  [yellow]•[/yellow] {note}")


def render_terminal_text(report: AnalysisReport, *, width: int = 100) -> str:
    """Render to a plain string (for tests / golden snapshots)."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
    )
    render_terminal(report, console=console)
    return buffer.getvalue()


def _print_profile_line(console: Console, report: AnalysisReport) -> None:
    metric = report.competition.get("metric")
    metric_label = ""
    if isinstance(metric, dict):
        metric_label = str(metric.get("name") or metric.get("label") or "")
    elif metric:
        metric_label = str(metric)
    problem = report.competition.get("problem_type") or ""
    bits = [bit for bit in (problem, metric_label) if bit]
    if bits:
        console.print(f"  [dim]{' · '.join(bits)}[/dim]")


def _print_winning_solutions(console: Console, report: AnalysisReport) -> None:
    winning = report.competition.get("winning_solutions")
    if not isinstance(winning, dict):
        console.print("    Status: [yellow]Unavailable[/yellow]")
        console.print("    Reason: Not available through configured provider.")
        return
    status = str(winning.get("status") or "unavailable")
    reason = str(winning.get("reason") or "Not available through configured provider.")
    if status == "ok" and winning.get("available"):
        items = winning.get("items") or []
        console.print(f"    Status: ok ({len(items)} items)")
        for item in items[:5]:
            if isinstance(item, dict):
                console.print(f"    • {item.get('title') or item.get('id')}")
            else:
                console.print(f"    • {item}")
        return
    console.print(f"    Status: [yellow]{status.title()}[/yellow]")
    console.print(f"    Reason: {reason}")


def _experiment_labels(report: AnalysisReport) -> list[str]:
    labels: list[str] = []
    for artifact in report.artifacts:
        if str(getattr(artifact, "type", "") or "") != "experiment":
            continue
        labels.append(artifact.title or artifact.id)
    for doc_id in report.retrieval.experiments:
        if doc_id and doc_id not in labels:
            labels.append(doc_id)
    return list(dict.fromkeys(labels))


def _failure_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for doc_id in report.retrieval.failures:
        if doc_id:
            lines.append(str(doc_id))
    for artifact in report.artifacts:
        if str(getattr(artifact, "type", "") or "") != "experiment":
            continue
        text = f"{artifact.title} {artifact.summary}".lower()
        if any(marker in text for marker in _FAILURE_MARKERS):
            lines.append(f"{artifact.id}: {artifact.summary or artifact.title}")
    return list(dict.fromkeys(lines))


def _opportunity_lines(report: AnalysisReport) -> list[str]:
    lines: list[str] = []
    for transfer in report.transfer_opportunities[:5]:
        hint = transfer.get("hypothesis_hint") or transfer.get("summary")
        if hint:
            lines.append(str(hint))
    for card in report.hypothesis_recommendations[:5]:
        title = card.get("title")
        if title and title not in lines:
            lines.append(str(title))
    return lines
