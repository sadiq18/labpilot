"""analyze_competition tool handler."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.artifacts.analysis import write_analysis
from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import build_default_registry
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def analyze_competition(
    workspace: Workspace,
    *,
    only: str | None = None,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    llm_client: Any | None = None,
    ingest_knowledge: bool = True,
    hypothesize: bool = True,
    brief: bool = True,
    fetch_kaggle: bool = False,
    refresh: bool = False,
) -> ToolResult:
    """Run competition analysis and persist ``analyze.json`` via the artifact adapter."""
    context = build_context(
        workspace.competition,
        runs_dir=workspace.effective_runs_dir,
        knowledge_dir=workspace.knowledge_dir,
        refresh=refresh,
    )
    orchestrator = AnalyzeOrchestrator(
        build_default_registry(),
        llm_client=llm_client,
        ingest_knowledge=ingest_knowledge,
        hypothesize=hypothesize,
        brief=brief,
        fetch_kaggle=fetch_kaggle,
    )
    report = orchestrator.analyze(
        context, only=only, include=include, exclude=exclude
    )
    ref = write_analysis(
        report,
        workspace.knowledge_dir,
        workspace.competition,
        path=context.report_path,
    )
    return ToolResult(
        refs=[ref],
        data={
            "competition": workspace.competition,
            "analyzers": list(report.analyzers),
            "path": ref.path,
        },
    )
