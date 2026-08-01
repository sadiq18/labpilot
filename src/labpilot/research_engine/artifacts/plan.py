"""Read/write adapters for research plans (DB plus optional file projections)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.planner.schemas.task_types import PlanStatus
from labpilot.research_engine.planner.serializer import write_projections
from labpilot.research_engine.planner.store import PlanStore

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["research_plan"]


class PlanArtifacts:
    """Typed access to research plans for one competition.

    The plan database is the source of record. When enabled, JSON/Markdown
    projections under ``research/plans/`` are written for inspection and diffs.
    """

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self._store = PlanStore(knowledge_dir, competition)
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()

    def close(self) -> None:
        """Release the underlying plan store connection."""
        self._store.close()

    @property
    def store(self) -> PlanStore:
        """Underlying :class:`PlanStore` for APIs not mirrored here (e.g. id allocation)."""
        return self._store

    def upsert(
        self,
        plan: ResearchPlan,
        *,
        write_projection_files: bool = True,
        produced_by: str = "plan",
    ) -> ArtifactRef:
        """Insert or replace a plan; optionally refresh JSON/MD projections.

        Returns an :class:`ArtifactRef` whose ``path`` points at the JSON
        projection when projections are written.
        """
        self._store.upsert_plan(plan)
        path: str | None = None
        if write_projection_files:
            write_projections(
                plan,
                knowledge_dir=self.knowledge_dir,
                competition=self.competition,
            )
            path = str(self.paths.plans_dir / f"{plan.id}.json")
        _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
        return ArtifactRef(
            kind="research_plan",
            id=plan.id,
            schema_id=SCHEMA_ID,
            path=path,
            competition=self.competition,
        )

    def get(self, plan_id: str) -> ResearchPlan | None:
        """Return a plan by id, or ``None`` if it does not exist."""
        return self._store.get_plan(plan_id)

    def list(
        self,
        *,
        status: PlanStatus | str | None = None,
        hypothesis_id: str | None = None,
    ) -> list[ResearchPlan]:
        """List plans, optionally filtered by status and/or hypothesis."""
        return self._store.list_plans(status=status, hypothesis_id=hypothesis_id)
