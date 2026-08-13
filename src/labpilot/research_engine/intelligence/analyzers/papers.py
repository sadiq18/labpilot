"""PaperAnalyzer — literature collect + PaperKnowledge extract (Plan 6).

Deterministic Engine owns search/cache via ``LiteratureProvider``. Micro Agent
fills ``PaperKnowledge`` (LLM optional, rule_engine fallback). Never “summarize
this paper.”
"""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.llm.client import LLMClient
from labpilot.research_engine.intelligence.analyzers.base import BaseAnalyzer
from labpilot.research_engine.intelligence.knowledge import KnowledgeStore
from labpilot.research_engine.intelligence.literature.models import Paper, PaperKnowledge
from labpilot.research_engine.intelligence.literature.provider import (
    ChainedLiteratureProvider,
    LiteratureProvider,
    literature_from_settings,
)
from labpilot.research_engine.intelligence.literature.query import build_literature_query
from labpilot.research_engine.intelligence.literature.ranking import (
    citation_velocity,
    select_for_extract,
    years_old,
)
from labpilot.research_engine.intelligence.micro_agents.paper_analyzer import (
    PaperAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)

logger = logging.getLogger("labpilot.research_engine.intelligence.analyzers.papers")

DEFAULT_EXTRACT_LIMIT = 15
DEFAULT_SEARCH_LIMIT = 40


class PaperAnalyzer(BaseAnalyzer):
    name = "papers"
    default_enabled = True

    def __init__(
        self,
        *,
        literature: LiteratureProvider | None = None,
        llm_client: LLMClient | None = None,
        extract_limit: int = DEFAULT_EXTRACT_LIMIT,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        competitions_dir: Path | None = None,
        persist: bool = True,
        download_pdfs: bool = True,
    ) -> None:
        self.literature = literature
        self.llm_client = llm_client
        self.extract_limit = extract_limit
        self.search_limit = search_limit
        self.competitions_dir = competitions_dir
        self.persist = persist
        self.download_pdfs = download_pdfs
        self._llm_explicit = llm_client is not None
        self._literature_explicit = literature is not None

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        self._maybe_attach_llm_client()
        notes: list[str] = []

        literature = self._resolve_literature(context)
        queries = build_literature_query(
            context,
            llm_client=self.llm_client,
            competitions_dir=self.competitions_dir,
        )
        notes.append(f"literature query: {queries!r}")

        try:
            papers = literature.search(
                queries, context=context, limit=self.search_limit
            )
        except Exception as exc:
            logger.warning("Literature search failed: %s", exc)
            notes.append(f"literature search: unavailable — {exc}")
            papers = []

        if isinstance(literature, ChainedLiteratureProvider):
            notes.extend(literature.notes)

        if not papers:
            return ResearchArtifacts(
                analyzer=self.name,
                items=[],
                notes=[*notes, "No papers found (search empty or soft-failed)."],
            )

        ranked_for_note = sorted(papers, key=lambda p: p.rank_score(), reverse=True)
        top = select_for_extract(papers, limit=self.extract_limit)
        n_recent = sum(1 for p in top if p.bucket() == "recent")
        n_found = len(top) - n_recent
        notes.append(
            f"Extracting PaperKnowledge for {len(top)} of {len(ranked_for_note)} "
            f"(limit={self.extract_limit}; recent={n_recent}, foundational={n_found})."
        )

        agent = PaperAnalyzerAgent(llm_client=self.llm_client)
        items: list[ResearchArtifact] = []
        techniques_rollup: list[str] = []
        llm_ok = 0
        rule_ok = 0
        for paper in top:
            knowledge, used_llm = self._extract(agent, context, paper)
            if used_llm:
                llm_ok += 1
            else:
                rule_ok += 1
            artifact = knowledge_to_artifact(context, paper, knowledge)
            items.append(artifact)
            techniques_rollup.extend(knowledge.techniques)

        if llm_ok and rule_ok:
            notes.append(
                f"paper extraction: mixed (llm={llm_ok}, rule_engine={rule_ok})."
            )
        elif llm_ok:
            notes.append(f"paper extraction: llm ({llm_ok}).")
        else:
            notes.append(f"paper extraction: rule_engine ({rule_ok}).")

        if self.persist:
            notes.extend(self._persist(context, items))

        return ResearchArtifacts(
            analyzer=self.name,
            items=items,
            notes=notes,
            techniques=list(dict.fromkeys(techniques_rollup)),
            opportunities=[
                f"paper:{a.id}" for a in items if a.techniques or a.claims
            ],
        )

    def _extract(
        self,
        agent: PaperAnalyzerAgent,
        context: AnalyzeContext,
        paper: Paper,
    ) -> tuple[PaperKnowledge, bool]:
        text = _paper_text_for_extract(paper)
        data = {
            "paper_id": paper.id,
            "title": paper.title,
            "datasets": paper.datasets,
            "datasets_used": paper.datasets,
            "code_urls": paper.github_urls,
            "github_urls": paper.github_urls,
            "benchmarks": paper.benchmarks,
            "grounded_in": "abstract" if paper.abstract else "metadata",
        }
        result = run_or_none(
            agent,
            StructuredContext(
                competition=context.competition,
                text=text,
                data=data,
            ),
        )
        if result is None:
            # Identity-only card — no invented extraction when the LLM fails.
            return (
                PaperKnowledge(
                    paper_id=paper.id,
                    title=paper.title,
                    datasets_used=list(paper.datasets),
                    code_urls=list(paper.github_urls),
                    grounded_in="abstract" if paper.abstract else "metadata",
                    confidence=0.2,
                ),
                False,
            )
        used_llm = bool(getattr(agent, "last_used_llm", False))
        if not isinstance(result, PaperKnowledge):
            result = PaperKnowledge.model_validate(result.model_dump())
        # Fill identity fields if the model omitted them.
        if not result.paper_id:
            result.paper_id = paper.id
        if not result.title:
            result.title = paper.title
        if not result.code_urls and paper.github_urls:
            result.code_urls = list(paper.github_urls)
        if not result.datasets_used and paper.datasets:
            result.datasets_used = list(paper.datasets)
        if paper.abstract:
            result.grounded_in = "abstract"
        return result, used_llm

    def _resolve_literature(self, context: AnalyzeContext) -> LiteratureProvider:
        if self.literature is not None:
            return self.literature
        return literature_from_settings(
            knowledge_dir=context.knowledge_dir,
            competition=context.competition,
            download_pdfs=self.download_pdfs,
        )

    def _maybe_attach_llm_client(self) -> None:
        if self._llm_explicit or self.llm_client is not None:
            return
        try:
            from labpilot.llm.client import resolve_llm_client
            from labpilot.workspace import load_config_for_cwd

            self.llm_client = resolve_llm_client(load_config_for_cwd()[0].llm)
        except Exception:
            self.llm_client = None

    def _persist(self, context: AnalyzeContext, items: list[ResearchArtifact]) -> list[str]:
        try:
            store = KnowledgeStore(context.knowledge_dir, context.competition)
            for artifact in items:
                store.upsert_artifact(artifact)
            return [f"Persisted {len(items)} paper artifact(s) to knowledge.db."]
        except Exception as exc:
            logger.warning("Paper artifact persist failed: %s", exc)
            return [f"persist: soft-fail — {exc}"]


def _paper_text_for_extract(paper: Paper) -> str:
    parts = [
        f"Title: {paper.title}",
        f"Authors: {', '.join(paper.authors)}" if paper.authors else "",
        f"Year: {paper.year}" if paper.year else "",
        f"Venue: {paper.venue}" if paper.venue else "",
        f"Citations: {paper.citations}" if paper.citations is not None else "",
        f"Concepts: {', '.join(paper.concepts)}" if paper.concepts else "",
        "",
        "Abstract:",
        paper.abstract or "(no abstract)",
    ]
    return "\n".join(p for p in parts if p is not None)


def knowledge_to_artifact(
    context: AnalyzeContext,
    paper: Paper,
    knowledge: PaperKnowledge,
) -> ResearchArtifact:
    summary_bits = knowledge.contributions[:2] or knowledge.methods[:1]
    summary = "; ".join(summary_bits) if summary_bits else paper.title
    return ResearchArtifact(
        id=f"paper:{paper.id}",
        type=ResearchArtifactType.PAPER,
        source="literature",
        title=paper.title or knowledge.title,
        summary=summary[:500],
        techniques=list(knowledge.techniques),
        models=[],
        datasets=list(knowledge.datasets_used or paper.datasets),
        claims=list(knowledge.contributions),
        references=list(paper.github_urls) + list(knowledge.code_urls),
        confidence=knowledge.confidence,
        competition_slug=context.competition,
        metadata={
            "paper": paper.model_dump(),
            "knowledge": knowledge.model_dump(),
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "citations": paper.citations,
            "relevance": paper.relevance,
            "rank_score": paper.rank_score(),
            "citation_velocity": citation_velocity(paper),
            "years_old": years_old(paper),
            "bucket": paper.bucket(),
            "ideas_worth_testing": knowledge.ideas_worth_testing,
            "limitations": knowledge.limitations,
            "methods": knowledge.methods,
            "grounded_in": knowledge.grounded_in,
            "feature_recipes": [
                recipe.model_dump(mode="json") for recipe in knowledge.feature_recipes
            ],
        },
    )


def paper_dict_for_report(artifact: ResearchArtifact) -> dict | None:
    """Compact paper card for ``AnalysisReport.papers`` (terminal Papers count)."""
    if artifact.type is not ResearchArtifactType.PAPER:
        return None
    meta = artifact.metadata or {}
    knowledge = meta.get("knowledge") if isinstance(meta.get("knowledge"), dict) else {}
    return {
        "id": artifact.id,
        "title": artifact.title,
        "summary": artifact.summary,
        "techniques": list(artifact.techniques),
        "claims": list(artifact.claims),
        "datasets": list(artifact.datasets),
        "confidence": artifact.confidence,
        "source": artifact.source,
        "arxiv_id": meta.get("arxiv_id"),
        "doi": meta.get("doi"),
        "citations": meta.get("citations"),
        "bucket": meta.get("bucket"),
        "grounded_in": meta.get("grounded_in") or knowledge.get("grounded_in"),
        "ideas_worth_testing": meta.get("ideas_worth_testing")
        or knowledge.get("ideas_worth_testing")
        or [],
        "methods": meta.get("methods") or knowledge.get("methods") or [],
        "limitations": meta.get("limitations") or knowledge.get("limitations") or [],
    }
