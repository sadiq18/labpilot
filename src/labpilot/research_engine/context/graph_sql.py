"""SQL-backed GraphPort — logical graph via competition knowledge DB."""

from __future__ import annotations

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


def default_graph_port(
    knowledge_dir: Path | None,
    competition: str,
) -> GraphPort:
    return SqlGraphPort(knowledge_dir=knowledge_dir, competition=competition)
