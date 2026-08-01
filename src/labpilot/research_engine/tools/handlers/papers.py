"""search_papers tool handler."""

from __future__ import annotations

import json
from typing import Any

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def search_papers(
    workspace: Workspace,
    *,
    query: str | None = None,
    limit: int = 10,
    offline: bool = False,
) -> ToolResult:
    """Search literature providers and write a papers projection under research/raw.

    When ``offline=True`` or the network/API fails, writes an empty hit list so
    the Conductor catalog still has a selectable tool.
    """
    q = (query or workspace.competition or "").strip()
    papers: list[dict[str, Any]] = []
    source = "offline"
    if not offline and q:
        try:
            from labpilot.research_engine.intelligence.literature.clients import (
                SemanticScholarClient,
            )

            client = SemanticScholarClient()
            found = client.search(q, limit=limit)
            papers = [
                {
                    "id": p.id,
                    "title": p.title,
                    "year": p.year,
                    "arxiv_id": p.arxiv_id,
                    "url": (p.urls or {}).get("semantic_scholar")
                    or (p.urls or {}).get("s2"),
                }
                for p in found
            ]
            source = "semantic_scholar"
        except Exception as exc:
            source = f"error:{type(exc).__name__}"

    out_dir = workspace.research_paths.raw_dir / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "search_papers.json"
    payload = {
        "query": q,
        "source": source,
        "count": len(papers),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    ref = ArtifactRef(
        kind="paper_search",
        id=f"papers:{workspace.competition}",
        schema_id="labpilot.artifact.paper_search/v1",
        path=str(path),
        competition=workspace.competition,
    )
    return ToolResult(refs=[ref], data=payload)
