"""Context Engine ports — providers and graph abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from labpilot.research_engine.context.models import ContextItem, ContextRequest


@runtime_checkable
class ContextProvider(Protocol):
    """Async retrieval source behind the Context Engine."""

    name: str

    async def fetch(self, request: ContextRequest) -> list[ContextItem]:
        """Return candidate items for this request (may be empty)."""


@runtime_checkable
class GraphPort(Protocol):
    """Abstract research-graph access (SQL-backed by default)."""

    def neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        limit: int = 20,
        hop_depth: int = 1,
    ) -> list[str]:
        """Return related node ids; empty if unknown."""

    def metrics_snapshot(self) -> GraphQueryMetrics:
        """Return counters for SQL-vs-graph-DB evaluation."""
