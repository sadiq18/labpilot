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
from labpilot.research_engine.context.providers.ri import RIRetrievalProvider

logger = logging.getLogger(__name__)


def default_providers(
    request: ContextRequest,
    *,
    llm_client: Any | None = None,
) -> list[ContextProvider]:
    """Default sources: RI retrieval. Additional providers register as needed."""
    _ = request
    return [RIRetrievalProvider(llm_client=llm_client)]


async def build_context_async(
    request: ContextRequest,
    *,
    providers: Sequence[ContextProvider] | None = None,
    graph: GraphPort | None = None,
    llm_client: Any | None = None,
) -> ContextBundle:
    """Gather providers concurrently and assemble a ContextBundle.

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

    items: list[ContextItem] = []
    for provider in active:
        items.extend(collected.get(provider.name, []))

    if request.max_items >= 0:
        items = items[: request.max_items]

    graph_metrics = _graph_metrics(graph)
    notes = [
        "identity assemble from providers",
        f"providers={[p.name for p in active]}",
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
