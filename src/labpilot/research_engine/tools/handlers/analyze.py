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


def _ensure_data_present(
    workspace: Workspace,
    kaggle_config: Any | None,
    on_progress: Callable[[str], None] | None,
) -> None:
    """Materialise ``data/raw`` before analysis, reusing the shared cache.

    Analysis (and therefore planning) is data-blind without this: the dataset
    profile used to appear only *after* a run, which is backwards from the
    order the planner needs it in. Soft-fails so analysis still runs offline.
    """
    raw_dir = workspace.raw_data_dir
    if raw_dir.is_dir() and any(p.is_file() for p in raw_dir.rglob("*")):
        return
    if kaggle_config is None:
        return
    from labpilot.diagnostics import kaggle_credentials_present

    if not kaggle_credentials_present():
        return
    try:
        from labpilot.accessor.data.downloader import DataDownloader

        if on_progress:
            on_progress("dataset: materialising data/raw (cache reuse if warm) …")
        DataDownloader(workspace.competition, kaggle_config).download(workspace.root)
        if on_progress:
            count = sum(1 for p in raw_dir.rglob("*") if p.is_file())
            on_progress(f"dataset: data/raw ready ({count} files)")
    # The Kaggle client calls sys.exit() on auth failure, which is SystemExit
    # (a BaseException) — catching only Exception would abort the whole command
    # with an empty stdout instead of degrading to offline analysis.
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        if on_progress:
            on_progress(f"dataset: download skipped ({exc})")


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
    on_progress: Callable[[str], None] | None = None,
    kaggle_config: Any | None = None,
) -> ToolResult:
    """Run competition analysis and persist ``analyze.json`` via the artifact adapter.

    Analyzers run first; ``verify_ai_artifact`` gates durable side effects
    (ingest / hypothesize / brief / fetch) and ``write_analysis``.
    ``reject`` skips those writes; ``spot_check`` writes with ``needs_review``.
    """
    competition_ref = url or workspace.competition
    _ensure_data_present(workspace, kaggle_config, on_progress)
    context = build_context(
        competition_ref,
        runs_dir=workspace.effective_runs_dir,
        knowledge_dir=workspace.knowledge_dir,
        refresh=refresh,
        data_dir=getattr(workspace, "raw_data_dir", None),
    )
    orchestrator = AnalyzeOrchestrator(
        build_default_registry(),
        llm_client=llm_client,
        ingest_knowledge=ingest_knowledge,
        hypothesize=hypothesize,
        brief=brief,
        fetch_kaggle=fetch_kaggle,
        on_progress=on_progress,
    )
    report = orchestrator.analyze_without_side_effects(
        context,
        only=only,
        # Conductor task args round-trip through JSON, where a set becomes a
        # list, so accept either shape.
        include=set(include) if include else None,
        exclude=set(exclude) if exclude else None,
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
                "needs_review": False,
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

    orchestrator.apply_side_effects(report, context)

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
