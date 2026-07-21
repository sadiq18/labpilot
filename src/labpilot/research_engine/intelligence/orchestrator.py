"""Thin orchestrator for ``research analyze`` (design §3).

Flow: select analyzers → run each (soft-fail) → merge ``ResearchArtifacts`` →
write the stub ``analyze.json``. Knowledge extraction, retrieval, and the
Hypothesis Assistant are wired in later plans; this stage only proves the
envelope and CLI contract.
"""

from __future__ import annotations

import logging

from labpilot.research_engine.intelligence.analyzers.base import Analyzer
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    AnalyzeContext,
    ResearchArtifacts,
)
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

logger = logging.getLogger("labpilot.research_engine.intelligence.orchestrator")


class AnalyzeOrchestrator:
    def __init__(self, registry: AnalyzerRegistry) -> None:
        self._registry = registry

    def analyze(
        self,
        context: AnalyzeContext,
        *,
        only: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> AnalysisReport:
        selected = self._registry.select(only=only, include=include, exclude=exclude)
        report = AnalysisReport(competition={"slug": context.competition})
        if context.url:
            report.competition["url"] = context.url

        if not selected:
            report.notes.append(
                "No analyzers selected/registered — wrote stub report only "
                "(real analyzers land in Plans 4–7)."
            )
            return report

        for analyzer in selected:
            emission = self._run_one(analyzer, context)
            report.analyzers.append(analyzer.name)
            report.artifacts.extend(emission.items)
            for note in emission.notes:
                report.notes.append(f"[{analyzer.name}] {note}")

        report.summary = {
            "analyzer_count": len(report.analyzers),
            "artifact_count": len(report.artifacts),
        }
        return report

    def _run_one(self, analyzer: Analyzer, context: AnalyzeContext) -> ResearchArtifacts:
        """Run a single analyzer, converting any exception into a soft-fail note."""
        try:
            return analyzer.analyze(context)
        except Exception as exc:  # soft-fail: one broken source must not abort the run
            logger.warning("Analyzer %r failed: %s", analyzer.name, exc)
            return ResearchArtifacts(
                analyzer=analyzer.name,
                notes=[f"analyzer failed: {exc}"],
            )
