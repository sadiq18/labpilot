"""Context Engine — orchestrate retrieve sources into ContextBundle.

Import direction::

    CLI / Conductor → context → {intelligence.retrieval, workspace, …}
    intelligence / tools must NOT import ``labpilot.research_engine.context``
"""

from __future__ import annotations

from labpilot.research_engine.context.bm25 import BM25, bm25_scores, tokenize
from labpilot.research_engine.context.compress import compress_candidates
from labpilot.research_engine.context.engine import build_context, build_context_async
from labpilot.research_engine.context.filters import apply_filters
from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from labpilot.research_engine.context.graph_sql import SqlGraphPort, default_graph_port
from labpilot.research_engine.context.models import ContextBundle, ContextItem, ContextRequest
from labpilot.research_engine.context.ports import ContextProvider, GraphPort
from labpilot.research_engine.context.providers import (
    EpisodicProvider,
    ExperimentProvider,
    RIRetrievalProvider,
    WorkspaceProvider,
)
from labpilot.research_engine.context.rank import rank_candidates
from labpilot.research_engine.context.retrieve import retrieve_candidates
from labpilot.research_engine.context.retrieve_metrics import Bm25RetrieveMetrics

__all__ = [
    "BM25",
    "Bm25RetrieveMetrics",
    "ContextBundle",
    "ContextItem",
    "ContextProvider",
    "ContextRequest",
    "EpisodicProvider",
    "ExperimentProvider",
    "GraphPort",
    "GraphQueryMetrics",
    "RIRetrievalProvider",
    "SqlGraphPort",
    "WorkspaceProvider",
    "apply_filters",
    "bm25_scores",
    "build_context",
    "build_context_async",
    "compress_candidates",
    "default_graph_port",
    "rank_candidates",
    "retrieve_candidates",
    "tokenize",
]
