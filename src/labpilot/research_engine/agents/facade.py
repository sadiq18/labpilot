"""Sync facade for specialist execute — Conductor stays sync."""

from __future__ import annotations

import anyio

from labpilot.research_engine.agents.ports import Agent
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.workspace_facade import Workspace


def execute_agent_sync(
    agent: Agent,
    task: object,
    workspace: Workspace,
    context: ContextBundle,
) -> list[ArtifactRef]:
    """Run ``agent.execute`` without requiring a caller-owned event loop."""

    async def _main() -> list[ArtifactRef]:
        return await agent.execute(task, workspace, context)

    return anyio.run(_main)
