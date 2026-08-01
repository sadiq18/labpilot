"""Capstone smoke: registry, events, parallel, git commit on experiment."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.agents import (
    EXPERIMENT_COMPLETED,
    AgentTask,
    EventBus,
    ParallelWorkItem,
    build_default_specialist_registry,
    execute_agent_sync,
    find_experiment_record,
    install_evidence_refresh_subscriber,
    run_parallel_sync,
)
from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.tools import build_default_tool_registry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import init_git_repo, scaffold_workspace


def _seed_plan(knowledge: Path, competition: str) -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="capstone",
            status=PlanStatus.READY,
            tasks=[
                ResearchTask(
                    id="P-001-T01",
                    plan_id="P-001",
                    type=TaskType.WRITE_CODE,
                    description="code",
                    order=0,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        store.upsert_plan(plan)
        return plan.id
    finally:
        store.close()


class _EchoAgent:
    name = "echo"
    capabilities = ["echo"]

    async def execute(self, task, workspace, context):  # noqa: ANN001
        del workspace, context
        tid = getattr(task, "id", "T")
        return [
            ArtifactRef(
                kind="echo",
                id=f"echo:{tid}",
                schema_id="labpilot.artifact.echo/v1",
                path=None,
                competition="cap",
            )
        ]


def test_m5_capstone_registry_events_parallel_git(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "m5-cap", "m5-cap")
    init_git_repo(client.root)
    ws = Workspace.from_client(client).ensure_roots()
    plan_id = _seed_plan(ws.knowledge_dir, ws.competition)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("baseline = True\n", encoding="utf-8")

    tools = build_default_tool_registry()
    assert {"implement", "run_experiment"} <= set(tools.names())

    bus = EventBus()
    seen: list[dict] = []
    bus.subscribe(EXPERIMENT_COMPLETED, lambda _e, p: seen.append(p))
    install_evidence_refresh_subscriber(bus)

    registry = build_default_specialist_registry(
        on_event=bus.publish,
        dry_run_default=True,
        install_subscribers=False,
    )
    # Implementation patches existing train and ensures infer.py
    impl_refs = execute_agent_sync(
        registry.require("implementation").agent,
        AgentTask(id="T-impl", capability="implement", description="layout"),
        ws,
        ContextBundle(request=ContextRequest(competition=ws.competition, goal="cap")),
    )
    assert impl_refs
    assert (ws.root / "pipeline" / "infer.py").is_file()

    exp_refs = execute_agent_sync(
        registry.require("experiment").agent,
        AgentTask(
            id="T-exp",
            capability="run_experiment",
            description="baseline",
            metadata={
                "plan_id": plan_id,
                "dry_run": True,
                "session_id": "S-001",
                "execution_id": "E-042",
            },
        ),
        ws,
        ContextBundle(request=ContextRequest(competition=ws.competition, goal="cap")),
    )
    assert exp_refs
    assert seen and seen[0].get("git_commit")
    record = find_experiment_record(ws.root, "E-042")
    assert record is not None
    assert record["git_commit"] == seen[0]["git_commit"]
    assert record.get("git_branch") == "research/S-001/E-042"
    note = ws.root / "artifacts" / f"evidence_refresh_{ws.competition}.json"
    assert note.is_file()

    # Thin parallel: ≥2 fake tasks under sync facade
    echo = _EchoAgent()
    parallel = run_parallel_sync(
        [
            ParallelWorkItem(id="p1", agent=echo, task=AgentTask(id="T1", capability="echo")),
            ParallelWorkItem(id="p2", agent=echo, task=AgentTask(id="T2", capability="echo")),
        ],
        ws,
        ContextBundle(request=ContextRequest(competition=ws.competition, goal="cap")),
        max_workers=2,
    )
    assert len(parallel) == 2 and all(r.ok for r in parallel)

    # Submit remains a gated catalog tool, not auto-invoked by experiment.
    assert tools.get("submit") is not None
    assert json.loads((ws.root / "experiment" / "record.json").read_text())["git_commit"]
