"""Thin orchestrator for ``research analyze`` (design §3).

Flow: select analyzers → run each (soft-fail) → optional Kaggle fetch →
upsert all artifacts → Knowledge Hub ingest → Hypothesis Assistant →
Research Brief → write ``analyze.json``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

from labpilot.research_engine.shared.experiments.models import HypothesisCreatedBy
from labpilot.research_engine.intelligence.analyzers.base import Analyzer
from labpilot.research_engine.intelligence.analyzers.competition import (
    profile_dict_for_report,
    related_dict_for_report,
)
from labpilot.research_engine.intelligence.analyzers.papers import paper_dict_for_report
from labpilot.research_engine.intelligence.analyzers.repositories import repo_dict_for_report
from labpilot.research_engine.intelligence.brief.builder import build_research_brief
from labpilot.research_engine.intelligence.fetch import KaggleFetchService
from labpilot.research_engine.intelligence.hypothesis import HypothesisAssistant
from labpilot.research_engine.intelligence.knowledge.hub import KnowledgeHub
from labpilot.research_engine.intelligence.knowledge.models import BeliefStatus, IngestResult
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    AnalyzeContext,
    ResearchArtifacts,
    ResearchArtifactType,
    TechniqueBuckets,
)
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry
from labpilot.research_engine.intelligence.repositories.local_profile import LocalCodeProfiler

logger = logging.getLogger("labpilot.research_engine.intelligence.orchestrator")

# Kernels and discussions are the highest-yield evidence on a Kaggle
# competition — they are where techniques, and therefore beliefs, claims and
# hypotheses, actually come from. Five of each produced too few concept
# candidates for the Knowledge Hub to propose anything to test.
# Override with LABPILOT_FETCH_KERNELS / LABPILOT_FETCH_DISCUSSIONS.
_FETCH_KERNEL_VOTE_LIMIT = int(os.environ.get("LABPILOT_FETCH_KERNELS", "15"))
_FETCH_KERNEL_SCORE_LIMIT = int(os.environ.get("LABPILOT_FETCH_KERNELS", "15"))
_FETCH_DISCUSSION_LIMIT = int(os.environ.get("LABPILOT_FETCH_DISCUSSIONS", "15"))


class AnalyzeOrchestrator:
    def __init__(
        self,
        registry: AnalyzerRegistry,
        *,
        llm_client: object | None = None,
        ingest_knowledge: bool = True,
        hypothesize: bool = True,
        brief: bool = True,
        fetch_kaggle: bool = False,
        kaggle_fetch_service: KaggleFetchService | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        self._registry = registry
        self._llm_client = llm_client
        self._ingest_knowledge = ingest_knowledge
        self._hypothesize = hypothesize
        self._brief = brief
        self._fetch_kaggle = fetch_kaggle
        self._kaggle_fetch_service = kaggle_fetch_service
        self._on_progress = on_progress

    def _progress(self, message: str) -> None:
        """Report a step boundary; analyzers can each take minutes on a local LLM."""
        if self._on_progress is not None:
            self._on_progress(message)

    def analyze(
        self,
        context: AnalyzeContext,
        *,
        only: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> AnalysisReport:
        report = self.analyze_without_side_effects(
            context, only=only, include=include, exclude=exclude
        )
        return self.apply_side_effects(report, context)

    def analyze_without_side_effects(
        self,
        context: AnalyzeContext,
        *,
        only: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> AnalysisReport:
        """Run analyzers only — no Kaggle fetch, hub ingest, hypotheses, or brief."""
        selected = self._registry.select(only=only, include=include, exclude=exclude)
        report = AnalysisReport(competition={"slug": context.competition})
        if context.url:
            report.competition["url"] = context.url

        if not selected and not self._fetch_kaggle:
            report.notes.append(
                "No analyzers selected/registered — wrote stub report only "
                "(real analyzers land in Plans 4–7)."
            )
            self._refresh_summary(report)
            return report

        total = len(selected)
        for index, analyzer in enumerate(selected, start=1):
            started = time.monotonic()
            self._progress(f"analyzer {index}/{total}: {analyzer.name} …")
            emission = self._run_one(analyzer, context)
            self._progress(
                f"analyzer {index}/{total}: {analyzer.name} done "
                f"({time.monotonic() - started:.1f}s, {len(emission.items)} artifacts)"
            )
            report.analyzers.append(analyzer.name)
            report.artifacts.extend(emission.items)
            for note in emission.notes:
                report.notes.append(f"[{analyzer.name}] {note}")
            self._merge_emission(report, emission)
            report.transfer_opportunities.extend(emission.transfers)

        self._refresh_summary(report)
        return report

    def apply_side_effects(
        self, report: AnalysisReport, context: AnalyzeContext
    ) -> AnalysisReport:
        """Run fetch / ingest / hypothesize / brief after an external verify gate."""
        for label, step in (
            ("kaggle fetch", self._fetch_kaggle_run),
            ("knowledge ingest", self._ingest),
            ("hypotheses", self._hypothesize_run),
            ("research brief", self._brief_run),
        ):
            started = time.monotonic()
            self._progress(f"{label} …")
            step(report, context)
            self._progress(f"{label} done ({time.monotonic() - started:.1f}s)")
        self._refresh_summary(report)
        return report

    def _refresh_summary(self, report: AnalysisReport) -> None:
        report.summary = {
            **dict(report.summary or {}),
            "analyzer_count": len(report.analyzers),
            "artifact_count": len(report.artifacts),
            "paper_count": len(report.papers),
            "repository_count": len(report.repositories),
            "transfer_count": len(report.transfer_opportunities),
            "knowledge_unit_count": len(report.knowledge_units),
            "hypothesis_count": len(report.hypothesis_recommendations),
            "has_research_brief": bool(report.research_brief),
        }

    def _fetch_kaggle_run(self, report: AnalysisReport, context: AnalyzeContext) -> None:
        """Opt-in: pull popular kernels + discussions before hub ingest."""
        if not self._fetch_kaggle:
            return
        try:
            service = self._kaggle_fetch_service or KaggleFetchService(
                llm_client=self._llm_client
            )
            calls: list[tuple[set[str], dict[str, Any]]] = [
                ({"kernels"}, {"kernel_sort": "voteCount", "limit": _FETCH_KERNEL_VOTE_LIMIT}),
                (
                    {"kernels"},
                    {"kernel_sort": "scoreDescending", "limit": _FETCH_KERNEL_SCORE_LIMIT},
                ),
                ({"discussions"}, {"limit": _FETCH_DISCUSSION_LIMIT}),
            ]
            fetched_ids: list[str] = []
            for sources, kwargs in calls:
                result = service.fetch(
                    context.competition,
                    sources=sources,  # type: ignore[arg-type]
                    knowledge_dir=context.knowledge_dir,
                    refresh=context.refresh,
                    **kwargs,
                )
                fetched_ids.extend(result.artifact_ids)
                report.notes.append(
                    f"[fetch-kaggle] sources={sorted(sources)} "
                    f"written={result.written} skipped={result.skipped_existing} "
                    f"fetched={result.fetched}"
                )
                for note in result.notes:
                    report.notes.append(f"[fetch-kaggle] {note}")

            if not fetched_ids:
                return
            with KnowledgeStore(context.knowledge_dir, context.competition) as store:
                existing_ids = {a.id for a in report.artifacts}
                for artifact_id in dict.fromkeys(fetched_ids):
                    if artifact_id in existing_ids:
                        continue
                    artifact = store.get_artifact(artifact_id)
                    if artifact is not None:
                        report.artifacts.append(artifact)
                        existing_ids.add(artifact_id)
        except Exception as exc:  # soft-fail
            logger.warning("Kaggle fetch during analyze failed: %s", exc)
            report.notes.append(f"[fetch-kaggle] failed: {exc}")

    def _ingest(self, report: AnalysisReport, context: AnalyzeContext) -> None:
        """Upsert analyzer artifacts, then hub-ingest (soft-fail)."""
        if not self._ingest_knowledge:
            report.notes.append("[knowledge-hub] ingestion skipped by request.")
            return
        if not report.artifacts and not self._fetch_kaggle:
            return
        try:
            with KnowledgeStore(context.knowledge_dir, context.competition) as store:
                for artifact in report.artifacts:
                    store.upsert_artifact(artifact)
                to_ingest = (
                    store.list_artifacts() if self._fetch_kaggle else list(report.artifacts)
                )
                if not to_ingest:
                    report.notes.append("[knowledge-hub] nothing to ingest.")
                    return
                result = KnowledgeHub(store, llm_client=self._llm_client).ingest(to_ingest)
                self._merge_knowledge(report, result)
                self._refresh_technique_buckets(report, store)
        except Exception as exc:  # soft-fail: merged knowledge is best-effort
            logger.warning("Knowledge hub ingest failed: %s", exc)
            report.notes.append(f"[knowledge-hub] ingest failed: {exc}")
            return

    def _hypothesize_run(self, report: AnalysisReport, context: AnalyzeContext) -> None:
        """Top-N recommendations only — soft-fail; never executes training."""
        if not self._hypothesize:
            report.notes.append("[hypothesis] skipped by request.")
            return
        pipeline = _pipeline_from_context(context)
        try:
            result = HypothesisAssistant(
                llm_client=self._llm_client,
                created_by=HypothesisCreatedBy.ANALYZE,
            ).recommend(
                knowledge_dir=context.knowledge_dir,
                competition=context.competition,
                question="Suggest next experiments for this competition",
                pipeline=pipeline,
                transfers=report.transfer_opportunities,
                persist=True,
                write_report=False,
                progressive=True,
            )
        except Exception as exc:  # soft-fail
            logger.warning("Hypothesis Assistant failed: %s", exc)
            report.notes.append(f"[hypothesis] failed: {exc}")
            return

        cards = [card.model_dump(mode="json") for card in result.recommendations]
        report.hypothesis_recommendations.extend(cards)
        report.suggested_experiments.extend(cards)
        report.hypotheses.extend(
            {
                "id": card.hypothesis_id,
                "title": card.title,
                "status": "proposed",
                "created_by": str(card.created_by),
                "generator": str(card.generator),
                "origin": str(card.origin),
            }
            for card in result.recommendations
            if card.hypothesis_id
        )
        for note in result.notes:
            report.notes.append(f"[hypothesis] {note}")
        if result.context is not None:
            report.retrieval.papers = [
                str(item.get("document_id") or item.get("label") or "")
                for item in result.context.papers
            ]
            report.retrieval.experiments = [
                str(item.get("document_id") or item.get("label") or "")
                for item in result.context.experiments
            ]
            report.retrieval.repositories = [
                str(item.get("document_id") or item.get("label") or "")
                for item in result.context.repositories
            ]
            report.retrieval.failures = [
                str(item.get("document_id") or item.get("label") or "")
                for item in result.context.failures
            ]

    def _brief_run(self, report: AnalysisReport, context: AnalyzeContext) -> None:
        """Durable Research Brief after ingest + hypothesize — soft-fail."""
        if not self._brief:
            report.notes.append("[research-brief] skipped by request.")
            return
        if not self._ingest_knowledge or not self._hypothesize:
            report.notes.append(
                "[research-brief] skipped (requires ingest + hypothesize)."
            )
            return
        try:
            with KnowledgeStore(context.knowledge_dir, context.competition) as store:
                brief = build_research_brief(
                    report, store, llm_client=self._llm_client
                )
            report.research_brief = brief.model_dump(mode="json")
            report.notes.append(f"[research-brief] generated_by={brief.generated_by}")
        except Exception as exc:  # soft-fail
            logger.warning("Research brief failed: %s", exc)
            report.notes.append(f"[research-brief] failed: {exc}")

    @staticmethod
    def _merge_knowledge(report: AnalysisReport, result: IngestResult) -> None:
        report.knowledge_units.extend(unit.model_dump(mode="json") for unit in result.units)
        for note in result.notes:
            report.notes.append(f"[knowledge-hub] {note}")
        for belief in result.beliefs:
            if belief.status is BeliefStatus.SUGGESTED:
                report.techniques.external_recommendations.append(belief.technique)
            elif belief.status in (BeliefStatus.VALIDATED, BeliefStatus.ESTABLISHED):
                report.techniques.locally_validated.append(belief.technique)
            elif belief.status is BeliefStatus.TESTING:
                report.techniques.unverified.append(belief.technique)

    @staticmethod
    def _refresh_technique_buckets(report: AnalysisReport, store: KnowledgeStore) -> None:
        """Rebuild technique buckets from durable beliefs (source of truth)."""
        external: list[str] = []
        local: list[str] = []
        unverified: list[str] = []
        for belief in store.list_beliefs():
            status = str(belief.get("status") or "").lower()
            technique = str(belief.get("technique") or "").strip()
            if not technique:
                continue
            if status == BeliefStatus.SUGGESTED:
                external.append(technique)
            elif status in (BeliefStatus.VALIDATED, BeliefStatus.ESTABLISHED):
                local.append(technique)
            elif status == BeliefStatus.TESTING:
                unverified.append(technique)
        report.techniques = TechniqueBuckets(
            external_recommendations=_unique(external),
            locally_validated=_unique(local),
            unverified=_unique(unverified),
        )

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
                report.competition = {**report.competition, **profile}
                continue
            related = related_dict_for_report(artifact)
            if related is not None:
                report.related_competitions.append(related)


def _pipeline_from_context(context: AnalyzeContext) -> list[str]:
    try:
        profile = LocalCodeProfiler().profile(context)
    except Exception:
        return []
    if profile is None:
        return []
    return list(
        dict.fromkeys(
            [
                *profile.architecture,
                *profile.augmentation,
                *profile.training_tricks,
                *profile.loss,
            ]
        )
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
