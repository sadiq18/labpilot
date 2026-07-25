"""Competition-level experiment report and HTML dashboard (Milestone 2, Plan 8)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from labpilot.experiments.graph import ExperimentGraph, build_graph
from labpilot.experiments.knowledge import KnowledgeBase
from labpilot.experiments.models import Experiment, ExperimentReport, RankedCandidate
from labpilot.experiments.ranking import (
    RankingWeights,
    rank_candidates,
    resolve_primary_metric_key,
)


class NoExperimentsError(ValueError):
    """Raised when a competition has zero assembled experiments."""


def build_report(
    competition: str,
    runs_dir: Path,
    knowledge_dir: Path,
    *,
    weights: RankingWeights | None = None,
    default_expected_gain: float = 0.0,
    cheap_tags: set[str] | None = None,
) -> ExperimentReport:
    """Compose Plans 1/3/5/6 into a single competition rollup. Raises if empty."""
    graph = build_graph(runs_dir, competition, knowledge_dir=knowledge_dir)
    if not graph.nodes:
        raise NoExperimentsError(
            f"No experiments yet for {competition}. "
            "Run `research run --competition ...` first."
        )

    metric_key = resolve_primary_metric_key(runs_dir, competition, graph)
    best_id, best_score = _pick_best_overall(graph, metric_key)

    kb = KnowledgeBase(knowledge_dir, competition)
    ranked = rank_candidates(
        competition,
        runs_dir,
        knowledge_dir,
        weights=weights,
        default_expected_gain=default_expected_gain,
        cheap_tags=cheap_tags,
    )

    best_pipeline: list[Experiment] = []
    if metric_key:
        best_pipeline = graph.best_path(metric_key)

    experiments = sorted(graph.nodes.values(), key=lambda e: e.created_at, reverse=True)
    return ExperimentReport(
        competition=competition,
        experiment_count=len(graph.nodes),
        primary_metric_key=metric_key,
        best_experiment_id=best_id,
        best_score=best_score,
        top_discoveries=kb.top_discoveries(3),
        known_failures=kb.known_failures(3),
        best_pipeline=best_pipeline,
        recommended_next=ranked[0] if ranked else None,
        experiments=experiments,
    )


def render_report_text(report: ExperimentReport, *, console: Console | None = None) -> None:
    """Print the brief mockup + all-experiments table via rich."""
    out = console or Console()
    metric = report.primary_metric_key or "metric"
    best = (
        f"{report.best_score:.4f}"
        if report.best_score is not None
        else "n/a"
    )

    out.print(f"\n[bold]{report.competition}[/bold]")
    out.print(f"{report.experiment_count} Experiments")
    out.print(f"Best ({metric}): {best}")
    if report.best_experiment_id:
        out.print(f"Best run: [cyan]{report.best_experiment_id}[/cyan]")

    out.print("\n[bold]Top Discoveries[/bold]")
    if report.top_discoveries:
        for entry in report.top_discoveries:
            out.print(f"  {entry.technique}  ({entry.delta_estimate:+.4f})")
    else:
        out.print("  (none yet)")

    out.print("\n[bold]Known Failures[/bold]")
    if report.known_failures:
        for entry in report.known_failures:
            out.print(f"  {entry.technique}  ({entry.delta_estimate:+.4f})")
    else:
        out.print("  (none yet)")

    out.print("\n[bold]Current Best Pipeline[/bold]")
    if report.best_pipeline:
        labels = [_pipeline_step_label(exp) for exp in report.best_pipeline]
        out.print("  " + " → ".join(labels))
    else:
        out.print("  (none)")

    out.print("\n[bold]Recommended Next[/bold]")
    _print_recommended(out, report.recommended_next)

    table = Table(title="All experiments (newest first)")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Progress")
    table.add_column("Description")
    table.add_column(metric)
    for exp in report.experiments:
        score = "-"
        if metric in exp.metrics:
            score = f"{exp.metrics[metric]:.4f}"
        desc = exp.description if len(exp.description) <= 56 else exp.description[:53] + "..."
        table.add_row(exp.id, exp.status, exp.progress, desc or "-", score)
    out.print(table)


def write_dashboard(
    report: ExperimentReport,
    graph: ExperimentGraph,
    *,
    knowledge_dir: Path,
    runs_dir: Path,
) -> Path:
    """Render HTML dashboard under knowledge/<slug>/dashboard.html."""
    html = render_dashboard_html(report, graph, runs_dir=runs_dir)
    out_dir = knowledge_dir / report.competition
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path


def render_dashboard_html(
    report: ExperimentReport,
    graph: ExperimentGraph,
    *,
    runs_dir: Path,
) -> str:
    """Build dashboard HTML string (same context shape as per-run reports)."""
    template_dir = Path(__file__).resolve().parent.parent / "report" / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        auto_reload=False,
    )
    metric_key = report.primary_metric_key
    tree_text = graph.to_tree_text(metric_key) if metric_key else graph.to_tree_text()

    # Relative from knowledge/<slug>/dashboard.html → runs/<id>/...
    def run_href(run_id: str, filename: str) -> str:
        return f"../../runs/{run_id}/{filename}"

    experiment_rows: list[dict[str, Any]] = []
    for exp in report.experiments:
        score = None
        if metric_key and metric_key in exp.metrics:
            score = exp.metrics[metric_key]
        report_rel = run_href(exp.id, "report.html")
        comparison_rel = None
        if exp.parent_id and (runs_dir / exp.id / "comparison.md").is_file():
            comparison_rel = run_href(exp.id, "comparison.md")
        experiment_rows.append(
            {
                "id": exp.id,
                "status": exp.status,
                "progress": exp.progress,
                "description": exp.description,
                "score": score,
                "created_at": exp.created_at.isoformat(),
                "report_href": report_rel,
                "comparison_href": comparison_rel,
                "has_report": (runs_dir / exp.id / "report.html").is_file(),
            }
        )

    recommended = None
    if report.recommended_next is not None:
        hyp = report.recommended_next.hypothesis
        recommended = {
            "id": hyp.id,
            "prediction": hyp.prediction,
            "confidence": hyp.confidence,
            "score": report.recommended_next.score,
        }

    context = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "competition": report.competition,
        "experiment_count": report.experiment_count,
        "primary_metric_key": metric_key or "metric",
        "best_experiment_id": report.best_experiment_id,
        "best_score": report.best_score,
        "top_discoveries": report.top_discoveries,
        "known_failures": report.known_failures,
        "best_pipeline_labels": [_pipeline_step_label(e) for e in report.best_pipeline],
        "recommended": recommended,
        "tree_text": tree_text,
        "experiments": experiment_rows,
    }
    return env.get_template("experiments_dashboard.html.j2").render(**context)


def _pick_best_overall(
    graph: ExperimentGraph, metric_key: str | None
) -> tuple[str | None, float | None]:
    if not metric_key:
        return None, None
    scored = [
        (exp.id, exp.metrics[metric_key])
        for exp in graph.nodes.values()
        if metric_key in exp.metrics
    ]
    if not scored:
        return None, None
    scored.sort(key=lambda item: item[1], reverse=graph.maximize)
    return scored[0]


def _pipeline_step_label(exp: Experiment) -> str:
    if exp.description.strip():
        return exp.description.strip()
    parts: list[str] = []
    if exp.template_name:
        parts.append(exp.template_name)
    parts.extend(exp.feature_recipes)
    return " + ".join(parts) if parts else exp.id


def _print_recommended(console: Console, candidate: RankedCandidate | None) -> None:
    if candidate is None:
        console.print("  (no proposed hypotheses — generate with `research hypothesize`)")
        return
    hyp = candidate.hypothesis
    console.print(f"  {hyp.prediction}")
    console.print(f"  Confidence: {hyp.confidence:.0%}  (score {candidate.score:.3f}, {hyp.id})")
