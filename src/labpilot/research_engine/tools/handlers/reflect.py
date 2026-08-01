"""reflect tool handler."""

from __future__ import annotations

from typing import Any

from labpilot.research_engine.artifacts.reflection import run_and_wrap
from labpilot.research_engine.tools.descriptors import ToolResult
from labpilot.research_engine.workspace_facade import Workspace


def reflect(
    workspace: Workspace,
    *,
    execution_id: str | None = None,
    workspace_path: str | None = None,
    plan_id: str | None = None,
    hypothesis_id: str | None = None,
    llm_client: Any | None = None,
    persist: bool = True,
) -> ToolResult:
    """Run reflection and return a typed result via the artifact adapter."""
    result, ref = run_and_wrap(
        workspace.knowledge_dir,
        workspace.competition,
        execution_id=execution_id,
        workspace_path=workspace_path or workspace.root,
        plan_id=plan_id,
        hypothesis_id=hypothesis_id,
        llm_client=llm_client,
        persist=persist,
        write_projection=persist,
    )
    return ToolResult(
        refs=[ref],
        data={
            "execution_id": result.execution_id,
            "evidence_id": result.evidence.get("id"),
            "belief_id": result.belief.get("belief_id"),
        },
    )
