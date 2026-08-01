"""Context Engine — sync facade over async AnyIO provider gather."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anyio

from labpilot.research_engine.context.compress import compress_candidates
from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from labpilot.research_engine.context.graph_sql import default_graph_port
from labpilot.research_engine.context.models import ContextBundle, ContextItem, ContextRequest
from labpilot.research_engine.context.ports import ContextProvider, GraphPort
from labpilot.research_engine.context.providers.episodic import EpisodicProvider
from labpilot.research_engine.context.providers.experiments import ExperimentProvider
from labpilot.research_engine.context.providers.ri import RIRetrievalProvider
from labpilot.research_engine.context.providers.workspace import WorkspaceProvider
from labpilot.research_engine.context.rank import rank_candidates
from labpilot.research_engine.context.retrieve import retrieve_candidates
from labpilot.research_engine.context.retrieve_metrics import Bm25RetrieveMetrics
from labpilot.research_engine.debug_metrics import emit_debug_metrics

logger = logging.getLogger(__name__)


def default_providers(
    request: ContextRequest,
    *,
    llm_client: Any | None = None,
) -> list[ContextProvider]:
    """Default sources: RI, workspace, experiments; episodic when knowledge is set."""
    providers: list[ContextProvider] = [RIRetrievalProvider(llm_client=llm_client)]
    if request.knowledge_dir is not None:
        providers.append(WorkspaceProvider())
        providers.append(ExperimentProvider())
        providers.append(EpisodicProvider())
    return providers


async def build_context_async(
    request: ContextRequest,
    *,
    providers: Sequence[ContextProvider] | None = None,
    graph: GraphPort | None = None,
    llm_client: Any | None = None,
) -> ContextBundle:
    """Gather providers → retrieve (BM25) → rank → compress → ContextBundle.

    On provider failure, log the error and continue with the rest.
    Rank expands via ``graph.neighbors`` so ``graph_metrics`` reflect real lookups.
    """
    active = list(providers) if providers is not None else default_providers(
        request, llm_client=llm_client
    )
    if graph is None:
        graph = default_graph_port(request.knowledge_dir, request.competition)

    collected: dict[str, list[ContextItem]] = {}
    errors: list[str] = []

    async def _run(provider: ContextProvider) -> None:
        try:
            collected[provider.name] = await provider.fetch(request)
        except Exception as exc:  # noqa: BLE001 — isolate provider faults
            msg = f"{provider.name}: {exc}"
            logger.warning("Context provider failed: %s", msg)
            errors.append(msg)
            collected[provider.name] = []

    async with anyio.create_task_group() as tg:
        for provider in active:
            tg.start_soon(_run, provider)

    raw: list[ContextItem] = []
    for provider in active:
        raw.extend(collected.get(provider.name, []))

    retrieved, bm25_metrics = retrieve_candidates(raw, request)
    ranked = rank_candidates(retrieved, request, graph=graph)
    items = compress_candidates(ranked, request)

    graph_metrics = _graph_metrics(graph)
    notes = [
        "pipeline: retrieve(BM25) → rank(rel+rec+graph) → compress",
        f"providers={[p.name for p in active]}",
        (
            f"raw={len(raw)} retrieved={len(retrieved)} "
            f"ranked={len(ranked)} kept={len(items)}"
        ),
        (
            f"budget max_items={request.max_items} "
            f"max_chars={request.max_chars} "
            f"max_item_chars={request.max_item_chars}"
        ),
        (
            f"bm25_top={bm25_metrics.top_score:.4f} "
            f"zero={bm25_metrics.scores_zero} "
            f"coverage={bm25_metrics.query_term_coverage:.2f} "
            f"low_top={bm25_metrics.low_top_score} "
            f"no_match={bm25_metrics.no_positive_match}"
        ),
        (
            f"graph neighbors={graph_metrics.neighbor_calls} "
            f"returned={graph_metrics.neighbor_nodes_returned} "
            f"empty={graph_metrics.neighbor_empty_results} "
            f"slow={graph_metrics.slow_queries} "
            f"errors={graph_metrics.errors} "
            f"latency_avg_ms={graph_metrics.neighbor_latency_ms_avg:.2f} "
            f"latency_max_ms={graph_metrics.neighbor_latency_ms_max:.2f}"
        ),
    ]
    _log_retrieve_metrics(
        competition=request.competition,
        providers=[p.name for p in active],
        raw_count=len(raw),
        kept=len(items),
        bm25=bm25_metrics,
        graph=graph_metrics,
    )
    return ContextBundle(
        request=request,
        items=items,
        provider_errors=errors,
        notes=notes,
        graph_metrics=graph_metrics,
        bm25_metrics=bm25_metrics,
    )


def _log_retrieve_metrics(
    *,
    competition: str,
    providers: list[str],
    raw_count: int,
    kept: int,
    bm25: Bm25RetrieveMetrics,
    graph: GraphQueryMetrics,
) -> None:
    """Emit retrieve metrics at debug; stdout only when LABPILOT_DEBUG_METRICS=1."""
    bm25_denom = bm25.scores_zero + bm25.scores_positive
    bm25_line = (
        f"bm25 top={bm25.top_score:.4f} second={bm25.second_score:.4f} "
        f"gap={bm25.score_gap:.4f} mean_kept={bm25.mean_kept_score:.4f} "
        f"zero={bm25.scores_zero}/{bm25_denom} "
        f"coverage={bm25.query_term_coverage:.2f} "
        f"tokens={bm25.query_token_count} "
        f"low_top={bm25.low_top_score} no_match={bm25.no_positive_match} "
        f"after_filter={bm25.candidates_after_filter}"
    )
    graph_line = (
        f"graph neighbors={graph.neighbor_calls} "
        f"returned={graph.neighbor_nodes_returned} "
        f"empty={graph.neighbor_empty_results} "
        f"slow={graph.slow_queries} errors={graph.errors} "
        f"latency_avg_ms={graph.neighbor_latency_ms_avg:.2f} "
        f"latency_max_ms={graph.neighbor_latency_ms_max:.2f} "
        f"hop_max={graph.hop_depth_requested_max}"
    )
    line = (
        f"[context] competition={competition} "
        f"providers={providers} raw={raw_count} kept={kept} | "
        f"{bm25_line} | {graph_line}"
    )
    emit_debug_metrics(logger, line)


def _graph_metrics(graph: GraphPort) -> GraphQueryMetrics:
    snap = getattr(graph, "metrics_snapshot", None)
    if callable(snap):
        return snap()
    return GraphQueryMetrics()


def build_context(
    request: ContextRequest,
    *,
    providers: Sequence[ContextProvider] | None = None,
    graph: GraphPort | None = None,
    llm_client: Any | None = None,
) -> ContextBundle:
    """Sync entry point for Conductor / CLI — runs AnyIO internally."""

    async def _main() -> ContextBundle:
        return await build_context_async(
            request,
            providers=providers,
            graph=graph,
            llm_client=llm_client,
        )

    return anyio.run(_main)
