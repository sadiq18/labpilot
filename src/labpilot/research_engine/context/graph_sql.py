"""SQL-backed GraphPort — logical graph via competition knowledge DB."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from labpilot.research_engine.context.graph_metrics import (
    GraphMetricsCollector,
    GraphQueryMetrics,
    timed_neighbor,
)
from labpilot.research_engine.context.ports import GraphPort

logger = logging.getLogger(__name__)


class SqlGraphPort:
    """Thin GraphPort over the competition knowledge DB.

    ``neighbors`` walks ``artifact_techniques`` + ``evidence_links`` (1-hop by
    default; BFS for ``hop_depth`` > 1). Every call records latency/result
    metrics for SQL-vs-graph-DB decisions.
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
            try:
                out = self._neighbors_impl(
                    node_id,
                    edge_types=edge_types,
                    limit=limit,
                    hop_depth=hop_depth,
                )
            except Exception:  # noqa: BLE001 — metrics must still record
                timer.error = True
                timer.result_count = 0
                logger.debug("SqlGraphPort.neighbors failed for %s", node_id, exc_info=True)
                return []
            timer.result_count = len(out)
            return out

    def _neighbors_impl(
        self,
        node_id: str,
        *,
        edge_types: list[str] | None,
        limit: int,
        hop_depth: int,
    ) -> list[str]:
        db_path = self._db_path()
        if db_path is None or not db_path.is_file():
            return []

        depth = max(1, int(hop_depth))
        cap = max(0, int(limit))
        if cap == 0:
            return []

        relations = {r for r in (edge_types or []) if r}
        found: list[str] = []
        seen: set[str] = {node_id}
        frontier = [node_id]

        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            for _hop in range(depth):
                next_frontier: list[str] = []
                for nid in frontier:
                    for neigh in self._one_hop(conn, nid, relations):
                        if neigh in seen:
                            continue
                        seen.add(neigh)
                        found.append(neigh)
                        next_frontier.append(neigh)
                        if len(found) >= cap:
                            return found[:cap]
                frontier = next_frontier
                if not frontier:
                    break
        return found[:cap]

    def _one_hop(
        self,
        conn: sqlite3.Connection,
        node_id: str,
        relations: set[str],
    ) -> list[str]:
        out: list[str] = []

        # artifact → techniques
        art_tech = conn.execute(
            """
            SELECT technique_id AS id, relation
            FROM artifact_techniques
            WHERE artifact_id = ?
            """,
            (node_id,),
        ).fetchall()
        for row in art_tech:
            if relations and row["relation"] not in relations:
                continue
            out.append(str(row["id"]))

        # technique → artifacts
        tech_art = conn.execute(
            """
            SELECT artifact_id AS id, relation
            FROM artifact_techniques
            WHERE technique_id = ?
            """,
            (node_id,),
        ).fetchall()
        for row in tech_art:
            if relations and row["relation"] not in relations:
                continue
            out.append(str(row["id"]))

        # evidence_links as source artifact
        as_src = conn.execute(
            """
            SELECT target_id AS id, relation
            FROM evidence_links
            WHERE artifact_id = ?
            """,
            (node_id,),
        ).fetchall()
        for row in as_src:
            if relations and row["relation"] not in relations:
                continue
            out.append(str(row["id"]))

        # evidence_links as target
        as_tgt = conn.execute(
            """
            SELECT artifact_id AS id, relation
            FROM evidence_links
            WHERE target_id = ? AND artifact_id IS NOT NULL AND artifact_id != ''
            """,
            (node_id,),
        ).fetchall()
        for row in as_tgt:
            if relations and row["relation"] not in relations:
                continue
            out.append(str(row["id"]))

        return out

    def _db_path(self) -> Path | None:
        if self.knowledge_dir is None or not self.competition:
            return None
        try:
            from labpilot.research_engine.intelligence.paths import ResearchPaths

            return ResearchPaths(Path(self.knowledge_dir), self.competition).db_path
        except Exception:  # noqa: BLE001
            return None

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
