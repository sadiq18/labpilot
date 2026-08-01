"""Unit tests for the specialist event bus."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.agents import (
    EVIDENCE_UPDATED,
    EXPERIMENT_COMPLETED,
    IMPLEMENTATION_FINISHED,
    AgentTask,
    EventBus,
    build_default_specialist_registry,
    execute_agent_sync,
    install_evidence_refresh_subscriber,
)
from labpilot.research_engine.conductor.policy import build_observe_bundle
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.workspace_facade import Workspace


def _ws(tmp_path: Path, slug: str = "bus") -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", slug, code_root=tmp_path / "ws"
    ).ensure_roots()


def _bundle(competition: str = "bus") -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition=competition, goal="test"))


def _seed_plan(knowledge: Path, competition: str) -> str:
    store = PlanStore(knowledge, competition)
    try:
        now = datetime.now(UTC)
        plan = ResearchPlan(
            id="P-001",
            competition=competition,
            hypothesis_id="",
            goal="mini",
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


def test_event_bus_publish_subscribe() -> None:
    bus = EventBus()
    seen: list[tuple[str, dict]] = []
    bus.subscribe("CustomEvent", lambda e, p: seen.append((e, p)))
    bus.publish("CustomEvent", {"x": 1})
    assert seen == [("CustomEvent", {"x": 1})]


def test_experiment_publishes_completed_with_refs(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "pub")
    plan_id = _seed_plan(ws.knowledge_dir, ws.competition)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("pass\n", encoding="utf-8")

    bus = EventBus()
    seen: list[tuple[str, dict]] = []
    bus.subscribe(EXPERIMENT_COMPLETED, lambda e, p: seen.append((e, p)))
    install_evidence_refresh_subscriber(bus)

    evidence_events: list[dict] = []
    bus.subscribe(EVIDENCE_UPDATED, lambda e, p: evidence_events.append(p))

    registry = build_default_specialist_registry(
        on_event=bus.publish,
        dry_run_default=True,
        install_subscribers=False,
    )
    agent = registry.require("experiment").agent
    refs = execute_agent_sync(
        agent,
        AgentTask(
            id="T-bus",
            capability="run_experiment",
            metadata={"plan_id": plan_id, "dry_run": True},
        ),
        ws,
        _bundle("pub"),
    )
    assert refs
    assert seen
    event_name, payload = seen[0]
    assert event_name == EXPERIMENT_COMPLETED
    assert payload.get("experiment_id")
    assert payload.get("execution_id")
    assert isinstance(payload.get("refs"), list) and payload["refs"]
    assert evidence_events
    assert evidence_events[0].get("observe_refresh") is True
    note = ws.root / "artifacts" / f"evidence_refresh_{ws.competition}.json"
    assert note.is_file()
    body = json.loads(note.read_text(encoding="utf-8"))
    assert body["experiment_id"] == payload["experiment_id"]


def test_observe_surfaces_evidence_refresh_without_replacing_decision_log(
    tmp_path: Path,
) -> None:
    ws = _ws(tmp_path, "obs")
    note_dir = ws.root / "artifacts"
    note_dir.mkdir(parents=True, exist_ok=True)
    note = note_dir / f"evidence_refresh_{ws.competition}.json"
    note.write_text(
        json.dumps(
            {
                "observe_refresh": True,
                "experiment_id": "exp_demo_E-1",
                "competition": ws.competition,
            }
        ),
        encoding="utf-8",
    )
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("goal")
        # Decision log remains independently readable.
        assert store.list_decisions(session.id) == []
        observe = build_observe_bundle(store, ws, session.id, include_context=False)
        assert observe["evidence_refresh"]["experiment_id"] == "exp_demo_E-1"
        assert store.list_decisions(session.id) == []
    finally:
        store.close()


def test_implementation_publishes_finished(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "impl")
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("x=1\n", encoding="utf-8")
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(IMPLEMENTATION_FINISHED, lambda e, _p: seen.append(e))
    registry = build_default_specialist_registry(on_event=bus.publish)
    execute_agent_sync(
        registry.require("implementation").agent,
        AgentTask(id="T-i", capability="implement"),
        ws,
        _bundle("impl"),
    )
    assert seen == [IMPLEMENTATION_FINISHED]


def test_experiment_module_does_not_import_reflection() -> None:
    import ast

    path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "labpilot"
        / "research_engine"
        / "agents"
        / "experiment.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "reflection" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "reflection" not in alias.name
