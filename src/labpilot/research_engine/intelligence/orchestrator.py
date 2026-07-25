"""Thin orchestrator for ``research analyze`` (design §3).

Flow: select analyzers → run each (soft-fail) → merge ``ResearchArtifacts`` →
ingest into the Knowledge Extraction hub once → write ``analyze.json``.
Retrieval and the Hypothesis Assistant are wired in later plans.

The hub runs **after** every analyzer so one failed source cannot leave a
partially merged knowledge base, and the orchestrator only depends on the
``KnowledgeHub.ingest`` contract — not on how units are stored or indexed.
"""

from __future__ import annotations

import logging

from labpilot.research_engine.intelligence.analyzers.base import Analyzer
from labpilot.research_engine.intelligence.analyzers.competition import (
    profile_dict_for_report,
    related_dict_for_report,
)
from labpilot.research_engine.intelligence.analyzers.papers import paper_dict_for_report
from labpilot.research_engine.intelligence.analyzers.repositories import repo_dict_for_report
from labpilot.research_engine.intelligence.knowledge.hub import KnowledgeHub
from labpilot.research_engine.intelligence.knowledge.models import BeliefStatus, IngestResult
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    AnalyzeContext,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

logger = logging.getLogger("labpilot.research_engine.intelligence.orchestrator")


class AnalyzeOrchestrator:
    def __init__(
        self,
        registry: AnalyzerRegistry,
        *,
        llm_client: object | None = None,
        ingest_knowledge: bool = True,
    ) -> None:
        self._registry = registry
        self._llm_client = llm_client
        self._ingest_knowledge = ingest_knowledge

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
            self._merge_emission(report, emission)
            report.transfer_opportunities.extend(emission.transfers)

        self._ingest(report, context)

        report.summary = {
            "analyzer_count": len(report.analyzers),
            "artifact_count": len(report.artifacts),
            "paper_count": len(report.papers),
            "repository_count": len(report.repositories),
            "transfer_count": len(report.transfer_opportunities),
            "knowledge_unit_count": len(report.knowledge_units),
        }
        return report

    def _ingest(self, report: AnalysisReport, context: AnalyzeContext) -> None:
        """Single end-of-run hub call; a hub failure must not lose the report."""
        if not self._ingest_knowledge:
            report.notes.append("[knowledge-hub] ingestion skipped by request.")
            return
        if not report.artifacts:
            return
        try:
            with KnowledgeStore(context.knowledge_dir, context.competition) as store:
                result = KnowledgeHub(store, llm_client=self._llm_client).ingest(
                    report.artifacts
                )
        except Exception as exc:  # soft-fail: merged knowledge is best-effort
            logger.warning("Knowledge hub ingest failed: %s", exc)
            report.notes.append(f"[knowledge-hub] ingest failed: {exc}")
            return
        self._merge_knowledge(report, result)

    @staticmethod
    def _merge_knowledge(report: AnalysisReport, result: IngestResult) -> None:
        report.knowledge_units.extend(unit.model_dump(mode="json") for unit in result.units)
        for note in result.notes:
            report.notes.append(f"[knowledge-hub] {note}")
        for belief in result.beliefs:
            # External reading is only ever a suggestion; local evidence moves a
            # technique to unverified-but-being-tested, never to validated here.
            if belief.status is BeliefStatus.SUGGESTED:
                report.techniques.external_recommendations.append(belief.technique)
            else:
                report.techniques.unverified.append(belief.technique)

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

    def _merge_emission(self, report: AnalysisReport, emission: ResearchArtifacts) -> None:
        """Fold analyzer emissions into typed report sections."""
        for artifact in emission.items:
            if artifact.type is ResearchArtifactType.PAPER:
                card = paper_dict_for_report(artifact)
                if card is not None:
                    report.papers.append(card)
                continue
            if artifact.type is ResearchArtifactType.REPOSITORY:
                card = repo_dict_for_report(artifact)
                if card is not None:
                    report.repositories.append(card)
                continue
            if artifact.type is not ResearchArtifactType.COMPETITION:
                continue
            profile = profile_dict_for_report(artifact)
            if profile is not None:
                # Keep the slug/url envelope keys; overlay the expert brief.
                report.competition = {**report.competition, **profile}
                continue
            related = related_dict_for_report(artifact)
            if related is not None:
                report.related_competitions.append(related)
