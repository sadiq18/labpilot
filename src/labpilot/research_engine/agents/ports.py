"""Ports for specialist agents and coding backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle
from labpilot.research_engine.workspace_facade import Workspace


@runtime_checkable
class Agent(Protocol):
    """Conductor-scheduled specialist — thin skill loop over tools/workspace."""

    name: str
    capabilities: list[str]

    async def execute(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        """Run one task; ``context`` is a required ContextBundle."""
        ...


@runtime_checkable
class CodingTool(Protocol):
    """Swappable coding backend for Implementation specialists."""

    async def implement(
        self,
        task: object,
        workspace: Workspace,
        context: ContextBundle,
    ) -> list[ArtifactRef]:
        """Propose/apply code changes; returns artifact refs for written paths."""
        ...
