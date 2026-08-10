"""A `TaskContext` for driving one capability directly — M20 exit criterion 2.

*Verification calls production, never resembles it.* A rejection test that
hand-builds the state a capability reads is a second implementation of the
context, free to drift from the real one exactly where it matters — which is how
the smoke gate came to run `python train.py` while training ran
`uv run --script`.

So this constructs the same `TaskContext` the engineer constructs, from the same
`ResearchPaths`, and nothing else.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import ResearchExecution
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType


def capability_context(
    tmp_path: Path,
    *,
    task_type: TaskType,
    competition: str = "demo",
    metadata: dict | None = None,
    constraints: dict | None = None,
) -> TaskContext:
    """A context pointed at a real workspace tree under `tmp_path`."""
    knowledge = tmp_path / "knowledge"
    paths = ResearchPaths(knowledge, competition).ensure()
    root = tmp_path / "competitions" / competition
    (root / "pipeline").mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    plan = ResearchPlan(
        id="P-001",
        competition=competition,
        hypothesis_id="",
        goal="prove the gate can say no",
        status=PlanStatus.READY,
        tasks=[
            ResearchTask(
                id="P-001-T01",
                plan_id="P-001",
                type=task_type,
                metadata=metadata or {},
            )
        ],
        created_at=now,
        updated_at=now,
        metadata={"plan_kind": "baseline"},
    )
    return TaskContext(
        plan=plan,
        task=plan.tasks[0],
        execution=ResearchExecution(id="E-001", plan_id="P-001", competition=competition),
        paths=paths,
        workspace_root=root,
        competition=competition,
        constraints=constraints or {},
    )
