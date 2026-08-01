"""SQL-backed GraphPort — logical graph via competition knowledge DB."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.context.graph_metrics import (
    GraphMetricsCollector,
    GraphQueryMetrics,
    timed_neighbor,
)
from labpilot.research_engine.context.ports import GraphPort


class SqlGraphPort:
    """Thin GraphPort over the competition knowledge DB.

    ``neighbors`` is a stub until graph expansion is wired through this port.
    Every call still records latency/result metrics for SQL-vs-graph-DB decisions.
    """

    name = "sql_graph"

    def __init__(
        self,
        knowledge_dir: Path | None = None,
        competition: str | None = None,
        *,
        metrics: GraphMetricsCollector | None = None,
    ) -> None:
        self.knowledge_dir = knowledge_dir
        self.competition = competition
        self.metrics = metrics or GraphMetricsCollector()

    def neighbors(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None = None,
        limit: int = 20,
        hop_depth: int = 1,
    ) -> list[str]:
        with timed_neighbor(self.metrics, hop_depth=hop_depth) as timer:
            # TODO(m4): query intelligence.graph for related node ids.
            _ = (node_id, edge_types, limit, self.knowledge_dir, self.competition)
            out: list[str] = []
            timer.result_count = len(out)
            return out

    def metrics_snapshot(self) -> GraphQueryMetrics:
        return self.metrics.copy()

    def log_metrics(self, *, prefix: str = "[graph]") -> None:
        """Emit neighbor metrics at debug (or stdout when LABPILOT_DEBUG_METRICS=1)."""
        from labpilot.research_engine.debug_metrics import emit_debug_metrics

        snap = self.metrics_snapshot()
        line = (
            f"{prefix} competition={self.competition} "
            f"neighbors={snap.neighbor_calls} returned={snap.neighbor_nodes_returned} "
            f"empty={snap.neighbor_empty_results} slow={snap.slow_queries} "
            f"errors={snap.errors} "
            f"latency_avg_ms={snap.neighbor_latency_ms_avg:.2f} "
            f"latency_max_ms={snap.neighbor_latency_ms_max:.2f} "
            f"hop_max={snap.hop_depth_requested_max}"
        )
        emit_debug_metrics(logging.getLogger(__name__), line)


def default_graph_port(
    knowledge_dir: Path | None,
    competition: str,
) -> GraphPort:
    return SqlGraphPort(knowledge_dir=knowledge_dir, competition=competition)
