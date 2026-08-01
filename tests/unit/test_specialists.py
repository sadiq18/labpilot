"""Tests for Implementation + Experiment specialists and Conductor routing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.agents import (
    AgentTask,
    build_default_specialist_registry,
    execute_agent_sync,
)
from labpilot.research_engine.conductor.actions import ResearchAction, map_research_action
from labpilot.research_engine.context.models import ContextBundle, ContextRequest
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask
from labpilot.research_engine.planner.schemas.task_types import PlanStatus, TaskType
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.tools import build_default_tool_registry
from labpilot.research_engine.workspace_facade import Workspace


def _bundle(competition: str = "demo") -> ContextBundle:
    return ContextBundle(request=ContextRequest(competition=competition, goal="test"))


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    return Workspace.from_competition(
        tmp_path / "knowledge", slug, code_root=tmp_path / "ws"
    ).ensure_roots()


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
                ResearchTask(
                    id="P-001-T02",
                    plan_id="P-001",
                    type=TaskType.RUN_TRAINING,
                    description="train",
                    dependencies=["P-001-T01"],
                    order=1,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
        store.upsert_plan(plan)
        return plan.id
    finally:
        store.close()


def test_default_specialist_registry_routes_impl_and_experiment() -> None:
    registry = build_default_specialist_registry()
    assert set(registry.names()) == {"experiment", "implementation"}
    impl = registry.candidates(capability="implement")
    assert len(impl) == 1 and impl[0].name == "implementation"
    exp = registry.candidates(capability="run_experiment")
    assert len(exp) == 1 and exp[0].name == "experiment"
    assert registry.candidates(capability="implement", budget=1.0) == []


def test_implementation_patches_existing_and_separates_infer(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    train = ws.root / "pipeline" / "train.py"
    train.parent.mkdir(parents=True, exist_ok=True)
    original = '"""existing train"""\nX = 42\n'
    train.write_text(original, encoding="utf-8")

    events: list[tuple[str, dict]] = []
    registry = build_default_specialist_registry(
        on_event=lambda e, p: events.append((e, p))
    )
    agent = registry.require("implementation").agent
    refs = execute_agent_sync(
        agent,
        AgentTask(id="T-patch", capability="implement", description="add features"),
        ws,
        _bundle(),
    )
    assert train.read_text(encoding="utf-8") == original
    infer = ws.root / "pipeline" / "infer.py"
    assert infer.is_file()
    assert any(r.path and r.path.endswith("infer.py") for r in refs)
    assert events and events[0][0] == "ImplementationFinished"
    assert events[0][1]["patched_existing"] is True


def test_implementation_greenfield_writes_train_and_infer(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    (ws.root / "profile.json").write_text(
        json.dumps(
            {
                "competition": "demo",
                "files": ["train.csv"],
                "train_file": "train.csv",
                "test_file": "test.csv",
                "sample_submission_file": "sample_submission.csv",
                "row_count": 10,
                "column_count": 2,
                "columns": [
                    {"name": "id", "dtype": "int"},
                    {"name": "target", "dtype": "int"},
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = build_default_specialist_registry()
    agent = registry.require("implementation").agent
    refs = execute_agent_sync(
        agent,
        AgentTask(
            id="T-new",
            capability="implement",
            description="baseline",
            metadata={"force_rewrite": True},
        ),
        ws,
        _bundle(),
    )
    assert (ws.root / "pipeline" / "train.py").is_file()
    assert (ws.root / "pipeline" / "infer.py").is_file()
    assert refs


def test_experiment_produces_metrics_artifact(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "exp")
    plan_id = _seed_plan(ws.knowledge_dir, ws.competition)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text(
        "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )
    (ws.root / "profile.json").write_text("{}", encoding="utf-8")

    events: list[tuple[str, dict]] = []
    registry = build_default_specialist_registry(
        dry_run_default=True,
        on_event=lambda e, p: events.append((e, p)),
    )
    agent = registry.require("experiment").agent
    refs = execute_agent_sync(
        agent,
        AgentTask(
            id="T-exp",
            capability="run_experiment",
            metadata={"plan_id": plan_id, "dry_run": True},
        ),
        ws,
        _bundle("exp"),
    )
    assert any(r.kind == "experiment" for r in refs)
    record = ws.root / "experiment" / "record.json"
    assert record.is_file()
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert "metrics" in payload
    assert payload["plan_id"] == plan_id
    assert events and events[-1][0] == "ExperimentCompleted"


def test_conductor_tools_route_to_specialists(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "route")
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("x = 1\n", encoding="utf-8")

    tools = build_default_tool_registry()
    assert "implement" in tools.names()
    assert "run_experiment" in tools.names()

    impl = tools.invoke("implement", ws, description="patch")
    assert impl.data["specialist"] == "implementation"
    assert (ws.root / "pipeline" / "infer.py").is_file()

    allow = set(tools.names())
    plan = map_research_action(
        ResearchAction(intent="implement feature engineering helpers"),
        allow,
    )
    assert not plan.unmapped
    assert plan.steps[0].tool == "implement"


def test_run_experiment_never_sets_submit(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "gate")
    plan_id = _seed_plan(ws.knowledge_dir, ws.competition)
    (ws.root / "pipeline").mkdir(parents=True, exist_ok=True)
    (ws.root / "pipeline" / "train.py").write_text("pass\n", encoding="utf-8")

    tools = build_default_tool_registry()
    result = tools.invoke(
        "run_experiment",
        ws,
        plan_id=plan_id,
        dry_run=True,
        submit=True,  # ignored / forced false
    )
    assert result.data["submit"] is False
    assert result.data["specialist"] == "experiment"


def test_specialists_do_not_import_each_other() -> None:
    """Implementation must not import Experiment and vice versa."""
    import ast

    agents = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "labpilot"
        / "research_engine"
        / "agents"
    )
    impl = (agents / "implementation.py").read_text(encoding="utf-8")
    exp = (agents / "experiment.py").read_text(encoding="utf-8")
    for tree, forbidden in (
        (ast.parse(impl), "experiment"),
        (ast.parse(exp), "implementation"),
    ):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert forbidden not in node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert forbidden not in alias.name
