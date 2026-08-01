"""Read/write adapters for plan executions (``E-xxx`` rows and layouts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.execution.schemas import ExecutionStatus, ResearchExecution
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.intelligence.paths import ResearchPaths

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["execution"]


class ExecutionArtifacts:
    """Typed access to research executions for one competition."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self._store = ExecutionStore(knowledge_dir, competition)
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()

    def close(self) -> None:
        """Release the underlying execution store connection."""
        self._store.close()

    @property
    def store(self) -> ExecutionStore:
        """Underlying :class:`ExecutionStore` for APIs not mirrored here."""
        return self._store

    def create(
        self,
        plan_id: str,
        *,
        workspace_path: str | None = None,
        runtime_target: str | None = None,
        metadata: dict[str, Any] | None = None,
        status: ExecutionStatus = "pending",
        produced_by: str = "run",
    ) -> tuple[ResearchExecution, ArtifactRef]:
        """Allocate a new execution for ``plan_id`` and return it with a ref."""
        execution = self._store.create_execution(
            plan_id,
            workspace_path=workspace_path,
            runtime_target=runtime_target,
            metadata=metadata,
            status=status,
        )
        ref = self._ref(execution, produced_by=produced_by)
        return execution, ref

    def get(self, execution_id: str) -> ResearchExecution | None:
        """Return an execution by id, or ``None`` if it does not exist."""
        return self._store.get_execution(execution_id)

    def update_status(
        self,
        execution_id: str,
        status: ExecutionStatus | str,
        **kwargs: Any,
    ) -> None:
        """Update execution status (and optional error / experiment fields)."""
        self._store.update_status(execution_id, status, **kwargs)

    def _ref(self, execution: ResearchExecution, *, produced_by: str) -> ArtifactRef:
        _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
        evidence_dir = self.paths.root / "executions" / execution.id
        return ArtifactRef(
            kind="execution",
            id=execution.id,
            schema_id=SCHEMA_ID,
            path=str(evidence_dir) if evidence_dir.exists() else None,
            competition=self.competition,
        )
