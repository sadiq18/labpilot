"""ContextBuilder — sole bridge from Knowledge Store → typed ResearchContext.

Pipeline (always): Intent → Symbolic → Rank(stub) → Expand → Compress → Validate.
Stage 3 embeddings are skipped in Plan 9 v1. Pipeline-diff is deferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.retrieval.compress import compress_bundle
from labpilot.research_engine.intelligence.retrieval.fetchers import SymbolicFetcher
from labpilot.research_engine.intelligence.retrieval.intent import classify_intent
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryPlan,
    QueryType,
    ResearchContext,
    RetrievalIntent,
    SymbolicBundle,
)
from labpilot.research_engine.intelligence.retrieval.plans import plan_for


class ContextBuilder:
    """Build prompt-ready ``ResearchContext`` without exposing the database."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        llm_client: object | None = None,
        fetcher: SymbolicFetcher | None = None,
    ) -> None:
        self.store = store
        self.llm_client = llm_client
        self.fetcher = fetcher or SymbolicFetcher(store)

    def build(
        self,
        query: str | RetrievalIntent = "",
        *,
        plan: QueryPlan | None = None,
        profile: dict[str, Any] | None = None,
        pipeline: list[str] | None = None,
        query_type: QueryType | str | None = None,
        competition: dict[str, Any] | None = None,
        constraints: list[str] | None = None,
        step: str | None = None,
    ) -> ResearchContext:
        intent = self._resolve_intent(
            query,
            profile=profile,
            pipeline=pipeline,
            query_type=query_type,
        )
        resolved_plan = plan or plan_for(intent.query_type)
        bundle = self.fetcher.fetch(intent, resolved_plan)
        bundle = self._rank_stub(bundle)  # Stage 3: identity pass-through
        cards, fields, brief, budget = compress_bundle(
            bundle,
            intent=intent,
            competition=competition
            or {"slug": self.store.competition, **(profile or {})},
            constraints=constraints,
        )
        notes = list(bundle.notes)
        notes.append(
            f"context: brief={budget['total_chars']}/{budget['total_budget']} chars "
            f"(dropped_sections={budget['dropped']})."
        )
        if not _brief_excludes_raw(brief):
            notes.append("context: validation warning — brief may contain raw dump markers.")

        return ResearchContext(
            competition=fields["competition"],
            experiments=fields["experiments"],
            techniques=fields["techniques"],
            papers=fields["papers"],
            repositories=fields["repositories"],
            failures=fields["failures"],
            constraints=fields["constraints"],
            question=fields["question"],
            step=step or (resolved_plan.rounds[-1] if resolved_plan.rounds else "compress"),
            intent=intent,
            brief=brief,
            budget=budget,
            notes=notes,
        )

    def _resolve_intent(
        self,
        query: str | RetrievalIntent,
        *,
        profile: dict[str, Any] | None,
        pipeline: list[str] | None,
        query_type: QueryType | str | None,
    ) -> RetrievalIntent:
        if isinstance(query, RetrievalIntent):
            intent = query.model_copy(deep=True)
            if pipeline and not intent.current_pipeline:
                intent.current_pipeline = list(pipeline)
            if query_type is not None:
                intent.query_type = QueryType(str(query_type))
            return intent
        return classify_intent(
            question=str(query or ""),
            profile=profile,
            pipeline=pipeline,
            query_type=query_type,
            llm_client=self.llm_client,
        )

    @staticmethod
    def _rank_stub(bundle: SymbolicBundle) -> SymbolicBundle:
        """Stage 3 Semantic Ranking — deferred; preserve symbolic order."""
        return bundle


def build_research_context(
    knowledge_dir: Path | str,
    competition: str,
    query: str | RetrievalIntent = "",
    *,
    llm_client: object | None = None,
    **kwargs: Any,
) -> ResearchContext:
    """Convenience: open store, build context, close store."""
    with KnowledgeStore(Path(knowledge_dir), competition) as store:
        return ContextBuilder(store, llm_client=llm_client).build(query, **kwargs)


def _brief_excludes_raw(brief: str) -> bool:
    """Guard: compressed brief must not look like raw PDF/thread dumps."""
    markers = ("\x00", "%PDF", "<html", "BEGIN THREAD", "full text:")
    lower = brief.lower()
    return not any(marker.lower() in lower for marker in markers)
