"""Metrics for SQL GraphPort — signals when a graph DB may be needed."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel

# Neighbor queries slower than this count as "slow" (ms).
SLOW_NEIGHBOR_MS = 50.0


class GraphQueryMetrics(BaseModel):
    """Counters from graph neighbor lookups during context build.

    Use these to decide whether logical SQL is enough or a graph DB (e.g. Kuzu)
    is justified: high latency, empty multi-hop results, or exploding fan-out.
    """

    neighbor_calls: int = 0
    neighbor_nodes_returned: int = 0
    neighbor_empty_results: int = 0
    neighbor_latency_ms_total: float = 0.0
    neighbor_latency_ms_max: float = 0.0
    slow_queries: int = 0
    errors: int = 0
    hop_depth_requested_max: int = 0

    @property
    def neighbor_latency_ms_avg(self) -> float:
        if self.neighbor_calls <= 0:
            return 0.0
        return self.neighbor_latency_ms_total / self.neighbor_calls

    def record_neighbor(
        self,
        *,
        latency_ms: float,
        result_count: int,
        hop_depth: int = 1,
        error: bool = False,
        slow_threshold_ms: float = SLOW_NEIGHBOR_MS,
    ) -> None:
        self.neighbor_calls += 1
        self.neighbor_latency_ms_total += max(0.0, latency_ms)
        self.neighbor_latency_ms_max = max(self.neighbor_latency_ms_max, latency_ms)
        self.hop_depth_requested_max = max(self.hop_depth_requested_max, hop_depth)
        if error:
            self.errors += 1
            return
        self.neighbor_nodes_returned += max(0, result_count)
        if result_count <= 0:
            self.neighbor_empty_results += 1
        if latency_ms >= slow_threshold_ms:
            self.slow_queries += 1


@dataclass
class GraphMetricsCollector:
    """Mutable collector attached to a GraphPort implementation."""

    snapshot: GraphQueryMetrics = field(default_factory=GraphQueryMetrics)
    slow_threshold_ms: float = SLOW_NEIGHBOR_MS

    def record_neighbor(
        self,
        *,
        latency_ms: float,
        result_count: int,
        hop_depth: int = 1,
        error: bool = False,
    ) -> None:
        self.snapshot.record_neighbor(
            latency_ms=latency_ms,
            result_count=result_count,
            hop_depth=hop_depth,
            error=error,
            slow_threshold_ms=self.slow_threshold_ms,
        )

    def copy(self) -> GraphQueryMetrics:
        return self.snapshot.model_copy(deep=True)


class timed_neighbor:
    """Context manager: time a neighbors() body and record metrics."""

    def __init__(
        self,
        collector: GraphMetricsCollector,
        *,
        hop_depth: int = 1,
    ) -> None:
        self._collector = collector
        self._hop_depth = hop_depth
        self._t0 = 0.0
        self.result_count = 0
        self.error = False

    def __enter__(self) -> timed_neighbor:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_exc: object) -> bool:
        latency_ms = (time.perf_counter() - self._t0) * 1000.0
        self._collector.record_neighbor(
            latency_ms=latency_ms,
            result_count=0 if exc_type else self.result_count,
            hop_depth=self._hop_depth,
            error=exc_type is not None or self.error,
        )
        return False
