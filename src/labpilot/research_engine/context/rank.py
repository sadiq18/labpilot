"""Rank ContextItems with documented relevance / recency / graph signals.

Signals (combined; weights below)::

    relevance  — BM25 / provider score from retrieve (normalized 0..1)
    recency    — exponential decay on metadata created_at / updated_at
    graph      — inverse hop distance via GraphPort.neighbors (seed=0)

Default weights: relevance 0.55, recency 0.20, graph 0.25.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from labpilot.research_engine.context.models import ContextItem, ContextRequest
from labpilot.research_engine.context.ports import GraphPort

logger = logging.getLogger(__name__)

# Documented blend for M4 rank (not learned).
WEIGHT_RELEVANCE = 0.55
WEIGHT_RECENCY = 0.20
WEIGHT_GRAPH = 0.25

# Recency half-life in days (score halves every N days).
RECENCY_HALF_LIFE_DAYS = 14.0

# How many top retrieve hits to expand via the graph.
GRAPH_SEED_TOP_K = 3
GRAPH_NEIGHBOR_LIMIT = 24
GRAPH_HOP_DEPTH = 1


def rank_candidates(
    items: list[ContextItem],
    request: ContextRequest,
    graph: GraphPort | None = None,
) -> list[ContextItem]:
    """Reorder candidates using relevance + recency + cheap graph distance."""
    if not items:
        return []

    distances = _graph_distances(items, graph) if graph is not None else {}
    max_rel = max((float(i.score) for i in items), default=0.0) or 1.0
    now = datetime.now(UTC)

    ranked: list[ContextItem] = []
    for item in items:
        relevance = float(item.score) / max_rel
        recency = _recency_score(item, now=now)
        dist = distances.get(_primary_node_id(item))
        if dist is None:
            # Also check alternate node ids for this item.
            dist = min(
                (distances[n] for n in _item_node_ids(item) if n in distances),
                default=None,
            )
        graph_score = 0.0 if dist is None else 1.0 / (1.0 + float(dist))
        final = (
            WEIGHT_RELEVANCE * relevance
            + WEIGHT_RECENCY * recency
            + WEIGHT_GRAPH * graph_score
        )
        reason = (
            f"{item.reason} | rank="
            f"rel={relevance:.3f},rec={recency:.3f},graph={graph_score:.3f}"
            f"(d={dist if dist is not None else 'na'}),final={final:.4f}"
        ).strip(" |")
        ranked.append(
            item.model_copy(
                update={
                    "score": float(final),
                    "reason": reason,
                    "metadata": {
                        **item.metadata,
                        "rank_relevance": relevance,
                        "rank_recency": recency,
                        "rank_graph": graph_score,
                        "rank_graph_distance": dist,
                        "rank_final": final,
                    },
                }
            )
        )

    ranked.sort(key=lambda i: i.score, reverse=True)
    return ranked


def _graph_distances(
    items: list[ContextItem],
    graph: GraphPort,
) -> dict[str, int]:
    """Map node_id → hop distance from top retrieve seeds (0 = seed)."""
    distances: dict[str, int] = {}
    seeds = sorted(items, key=lambda i: i.score, reverse=True)[:GRAPH_SEED_TOP_K]
    seed_nodes: list[str] = []
    for item in seeds:
        for nid in _item_node_ids(item):
            if nid not in distances:
                distances[nid] = 0
                seed_nodes.append(nid)

    for nid in seed_nodes:
        try:
            neighbors = graph.neighbors(
                nid,
                limit=GRAPH_NEIGHBOR_LIMIT,
                hop_depth=GRAPH_HOP_DEPTH,
            )
        except Exception as exc:  # noqa: BLE001 — isolate graph faults
            logger.warning("graph.neighbors failed for %s: %s", nid, exc)
            continue
        for neigh in neighbors:
            key = str(neigh)
            if key not in distances:
                distances[key] = GRAPH_HOP_DEPTH
            else:
                distances[key] = min(distances[key], GRAPH_HOP_DEPTH)
    return distances


def _primary_node_id(item: ContextItem) -> str | None:
    ids = _item_node_ids(item)
    return ids[0] if ids else None


def _item_node_ids(item: ContextItem) -> list[str]:
    """Best-effort graph node ids from metadata and item id."""
    meta = item.metadata or {}
    found: list[str] = []
    for key in (
        "node_id",
        "artifact_id",
        "technique_id",
        "card_id",
        "hypothesis_id",
        "document_id",
    ):
        val = meta.get(key)
        if val:
            found.append(str(val))

    raw = meta.get("raw")
    if isinstance(raw, dict):
        for key in ("id", "document_id", "technique_id"):
            val = raw.get(key)
            if val:
                found.append(str(val))

    # Strip provider prefixes: "experiments:artifact:exp:…" → "exp:…"
    parts = item.id.split(":", 2)
    if len(parts) == 3 and parts[0] in {
        "experiments",
        "ri_retrieval",
        "workspace",
        "episodic",
    }:
        found.append(parts[2])
    elif len(parts) >= 2:
        found.append(parts[-1])

    # Deduplicate, preserve order.
    out: list[str] = []
    seen: set[str] = set()
    for nid in found:
        if nid and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def _recency_score(item: ContextItem, *, now: datetime) -> float:
    ts = _parse_timestamp(item.metadata)
    if ts is None:
        return 0.5
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


def _parse_timestamp(metadata: dict[str, Any]) -> datetime | None:
    for key in ("updated_at", "created_at", "timestamp", "built_at"):
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            text = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    return None
