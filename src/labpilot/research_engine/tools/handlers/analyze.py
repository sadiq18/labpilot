"""analyze_competition tool handler."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from labpilot.research_engine.artifacts.analysis import write_analysis
from labpilot.research_engine.intelligence.context import build_context
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import build_default_registry
from labpilot.research_engine.shared.verify_artifact import (
    VerifyPrompt,
    VerifyResult,
    verify_ai_artifact,
)
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
    url: str | None = None,
    verify_auto: bool = True,
    verify: VerifyPrompt | Callable[[str, dict[str, Any]], VerifyResult] | None = None,
) -> ToolResult:
    """Run competition analysis and persist ``analyze.json`` via the artifact adapter.

    Before durable write, runs ``verify_ai_artifact`` (default auto-approve).
    ``reject`` skips the write; ``spot_check`` writes with ``needs_review``.
    """
    competition_ref = url or workspace.competition
    context = build_context(
        competition_ref,
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
    verification = verify_ai_artifact(
        "analysis_report",
        {
            "competition": workspace.competition,
            "analyzers": list(report.analyzers),
            "artifact_count": len(report.artifacts),
        },
        auto=verify_auto,
        prompt=verify,
    )
    if verification.decision == "reject":
        return ToolResult(
            refs=[],
            data={
                "competition": workspace.competition,
                "analyzers": list(report.analyzers),
                "path": None,
                "brief_path": str(context.paths.brief_path),
                "report": report,
                "verification": verification.model_dump(),
                "written": False,
            },
        )

    if verification.decision == "spot_check":
        summary = dict(report.summary or {})
        summary["needs_review"] = True
        summary["verification"] = verification.model_dump()
        report.summary = summary
        note = "needs_review: spot_check"
        if note not in report.notes:
            report.notes = [*report.notes, note]

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
            "brief_path": str(context.paths.brief_path),
            "report": report,
            "verification": verification.model_dump(),
            "written": True,
            "needs_review": verification.decision == "spot_check",
        },
    )
