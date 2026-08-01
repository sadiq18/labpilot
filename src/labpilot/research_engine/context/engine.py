"""Context Engine — sync facade over async AnyIO provider gather."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import anyio

from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from labpilot.research_engine.context.graph_sql import default_graph_port
from labpilot.research_engine.context.models import ContextBundle, ContextItem, ContextRequest
from labpilot.research_engine.context.ports import ContextProvider, GraphPort
from labpilot.research_engine.context.providers.episodic import EpisodicProvider
from labpilot.research_engine.context.providers.experiments import ExperimentProvider
from labpilot.research_engine.context.providers.ri import RIRetrievalProvider
from labpilot.research_engine.context.providers.workspace import WorkspaceProvider
from labpilot.research_engine.context.retrieve import retrieve_candidates

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
    """Gather providers concurrently, filter, BM25-score, and assemble a bundle.

    On provider failure, log the error and continue with the rest.
    """
    active = list(providers) if providers is not None else default_providers(
        request, llm_client=llm_client
    )
    if graph is None:
        graph = default_graph_port(request.knowledge_dir, request.competition)
    # TODO(m4): use graph.neighbors for expand/rank (graph distance / related nodes).
    # Neighbor calls should record into graph.metrics_snapshot() for SQL-vs-Kuzu signals.

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

    items = retrieve_candidates(raw, request)

    graph_metrics = _graph_metrics(graph)
    notes = [
        "retrieve: filter + BM25",
        f"providers={[p.name for p in active]}",
        f"raw={len(raw)} kept={len(items)}",
        (
            f"graph_neighbors={graph_metrics.neighbor_calls} "
            f"slow={graph_metrics.slow_queries} "
            f"empty={graph_metrics.neighbor_empty_results}"
        ),
    ]
    return ContextBundle(
        request=request,
        items=items,
        provider_errors=errors,
        notes=notes,
        graph_metrics=graph_metrics,
    )


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
