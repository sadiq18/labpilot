"""query_memory tool handler."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.intelligence.graph.query import query_techniques
from labpilot.research_engine.intelligence.retrieval.context_builder import (
    build_research_context,
)
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def query_memory(
    workspace: Workspace,
    *,
    query: str = "",
    llm_client: Any | None = None,
    include_techniques: bool = True,
    min_technique_confidence: float = 0.0,
) -> ToolResult:
    """Retrieve a research context bundle (and optional technique hits)."""
    context = build_research_context(
        workspace.knowledge_dir,
        workspace.competition,
        query,
        llm_client=llm_client,
    )
    payload: dict[str, Any]
    if hasattr(context, "model_dump"):
        payload = context.model_dump(mode="json")
    else:
        payload = {"repr": str(context)}

    techniques: list[dict[str, Any]] = []
    if include_techniques:
        techniques = query_techniques(
            knowledge_dir=workspace.knowledge_dir,
            competition=workspace.competition,
            min_confidence=min_technique_confidence,
        )

    ref = ArtifactRef(
        kind="memory_query",
        id=f"memory:{workspace.competition}",
        schema_id="labpilot.artifact.memory_query/v1",
        path=None,
        competition=workspace.competition,
    )
    return ToolResult(
        refs=[ref],
        data={
            "competition": workspace.competition,
            "query": query,
            "context": payload,
            "techniques": techniques,
        },
    )
