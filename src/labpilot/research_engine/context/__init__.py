"""Context Engine — orchestrate retrieve sources into ContextBundle.

Import direction::

    CLI / Conductor → context → {intelligence.retrieval, workspace, …}
    intelligence / tools must NOT import ``labpilot.research_engine.context``
"""

from __future__ import annotations

from labpilot.research_engine.context.engine import build_context, build_context_async
from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics
from labpilot.research_engine.context.graph_sql import SqlGraphPort, default_graph_port
from labpilot.research_engine.context.models import ContextBundle, ContextItem, ContextRequest
from labpilot.research_engine.context.ports import ContextProvider, GraphPort
from labpilot.research_engine.context.providers import RIRetrievalProvider

__all__ = [
    "ContextBundle",
    "ContextItem",
    "ContextProvider",
    "ContextRequest",
    "GraphPort",
    "GraphQueryMetrics",
    "RIRetrievalProvider",
    "SqlGraphPort",
    "build_context",
    "build_context_async",
    "default_graph_port",
]
