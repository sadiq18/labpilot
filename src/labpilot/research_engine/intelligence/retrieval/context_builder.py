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
        progressive: bool = False,
        core_technique_limit: int = 8,
    ) -> ResearchContext:
        intent = self._resolve_intent(
            query,
            profile=profile,
            pipeline=pipeline,
            query_type=query_type,
        )
        resolved_plan = plan or plan_for(intent.query_type)
        competition_meta = competition or {
            "slug": self.store.competition,
            **(profile or {}),
        }

        if progressive:
            return self._build_progressive(
                intent,
                resolved_plan,
                competition=competition_meta,
                constraints=constraints,
                core_technique_limit=core_technique_limit,
            )

        bundle = self.fetcher.fetch(intent, resolved_plan)
        bundle = self._rank_stub(bundle)  # Stage 3: identity pass-through
        return self._finalize(
            bundle,
            intent=intent,
            plan=resolved_plan,
            competition=competition_meta,
            constraints=constraints,
            step=step,
        )

    def _build_progressive(
        self,
        intent: RetrievalIntent,
        plan: QueryPlan,
        *,
        competition: dict[str, Any],
        constraints: list[str] | None,
        core_technique_limit: int,
    ) -> ResearchContext:
        """Fixed 3-pass flow: core techniques → expand survivors → compress."""
        # Pass 1 — core selection (L2 technique metadata only).
        core_plan = plan.model_copy(deep=True)
        core_plan.limits = {
            **core_plan.limits,
            "techniques": core_technique_limit,
            "papers": 0,
            "experiments": 0,
            "repositories": 0,
            "failures": 0,
        }
        core = self.fetcher.fetch(intent, core_plan, expand=False)
        survivor_ids = [str(row["id"]) for row in core.techniques[:core_technique_limit]]

        # Pass 2 — targeted expansion for survivors only.
        expanded = self.fetcher.fetch(
            intent,
            plan,
            technique_ids=survivor_ids or None,
            expand=True,
        )
        expanded = self._rank_stub(expanded)

        # Pass 3 — budget compression.
        context = self._finalize(
            expanded,
            intent=intent,
            plan=plan,
            competition=competition,
            constraints=constraints,
            step="progressive_compress",
        )
        context.notes = [
            f"progressive: pass1_core={len(core.techniques)} "
            f"survivors={len(survivor_ids)}.",
            *core.notes,
            *context.notes,
        ]
        return context

    def _finalize(
        self,
        bundle: SymbolicBundle,
        *,
        intent: RetrievalIntent,
        plan: QueryPlan,
        competition: dict[str, Any],
        constraints: list[str] | None,
        step: str | None,
    ) -> ResearchContext:
        cards, fields, brief, budget = compress_bundle(
            bundle,
            intent=intent,
            competition=competition,
            constraints=constraints,
        )
        notes = list(bundle.notes)
        notes.append(
            f"context: brief={budget['total_chars']}/{budget['total_budget']} chars "
            f"(dropped_sections={budget['dropped']})."
        )
        if not _brief_excludes_raw(brief):
            notes.append(
                "context: validation warning — brief may contain raw dump markers."
            )

        return ResearchContext(
            competition=fields["competition"],
            experiments=fields["experiments"],
            techniques=fields["techniques"],
            papers=fields["papers"],
            repositories=fields["repositories"],
            failures=fields["failures"],
            constraints=fields["constraints"],
            question=fields["question"],
            step=step or (plan.rounds[-1] if plan.rounds else "compress"),
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
