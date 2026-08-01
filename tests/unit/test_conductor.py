"""Unit tests for Conductor store, scheduler, policy, approvals, loop."""

from __future__ import annotations

import ast
from pathlib import Path

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.approvals import maybe_approve
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.models import ApprovalResult, NextAction
from labpilot.research_engine.conductor.policy import (
    offline_next_action,
    validate_next_action,
)
from labpilot.research_engine.conductor.scheduler import Scheduler
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.handlers.papers import search_papers
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "demo") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def echo(workspace: Workspace, *, note: str = "ok", **_: object) -> ToolResult:
        path = workspace.artifacts_dir / "echo.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note, encoding="utf-8")
        return ToolResult(
            refs=[
                ArtifactRef(
                    kind="echo",
                    id="echo:1",
                    schema_id="labpilot.artifact.echo/v1",
                    path=str(path),
                    competition=workspace.competition,
                )
            ],
            data={"note": note},
        )

    def papers(workspace: Workspace, **kwargs: object) -> ToolResult:
        return search_papers(workspace, offline=True, query="demo")

    reg.register(ToolDescriptor(name="analyze_competition", handler=echo))
    reg.register(ToolDescriptor(name="search_papers", handler=papers))
    reg.register(ToolDescriptor(name="query_memory", handler=echo))
    return reg


def test_session_task_decision_round_trip(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("beat baseline")
        assert session.id.startswith("S-")
        task = store.enqueue(session.id, "search_papers", args={"offline": True})
        assert task.status == "pending"
        store.update_task_status(task.id, "running")
        store.update_task_status(
            task.id,
            "completed",
            artifact_refs=[{"kind": "paper_search", "id": "x"}],
        )
        got = store.get_task(task.id)
        assert got is not None
        assert got.status == "completed"
        assert got.artifact_refs[0]["kind"] == "paper_search"

        did = store.new_decision_id()
        from labpilot.research_engine.conductor.models import DecisionRecord

        store.append_decision(
            DecisionRecord(
                id=did,
                session_id=session.id,
                tool_name="search_papers",
                rationale="try papers",
            )
        )
        # Resume from DB
        store2 = ConductorStore(ws.knowledge_dir, ws.competition)
        try:
            assert store2.get_session(session.id) is not None
            assert len(store2.list_tasks(session.id)) == 1
            assert len(store2.list_decisions(session.id)) == 1
        finally:
            store2.close()
    finally:
        store.close()


def test_ready_tasks_respect_dependencies(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "deps")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("goal")
        t1 = store.enqueue(session.id, "analyze_competition")
        t2 = store.enqueue(
            session.id, "search_papers", dependencies=[t1.id], args={"offline": True}
        )
        ready = store.ready_tasks(session.id)
        assert [t.id for t in ready] == [t1.id]
        store.update_task_status(t1.id, "completed")
        ready2 = store.ready_tasks(session.id)
        assert [t.id for t in ready2] == [t2.id]
    finally:
        store.close()


def test_scheduler_dispatches_tool(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "sched")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _echo_registry()
    try:
        session = store.create_session("g")
        task = store.enqueue(session.id, "analyze_competition", args={"note": "hi"})
        sched = Scheduler(store, reg, ws)
        result = sched.dispatch(task)
        assert result.data["note"] == "hi"
        assert store.get_task(task.id).status == "completed"
    finally:
        store.close()


def test_policy_allowlist_and_offline_order() -> None:
    allow = {"analyze_competition", "search_papers", "query_memory"}
    bad = validate_next_action(
        NextAction(tool="invent_agent", rationale="nope"), allow
    )
    assert bad.stop and bad.tool is None
    observe = {"completed_tools": [], "operator_feedback": []}
    action = offline_next_action(observe, allow)
    assert action.tool == "analyze_competition"
    observe2 = {
        "completed_tools": ["analyze_competition"],
        "operator_feedback": [],
    }
    action2 = offline_next_action(observe2, allow)
    assert action2.tool == "search_papers"


def test_approval_reject_persists_feedback(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "appr")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")

        def reject(tool: str) -> ApprovalResult:
            return ApprovalResult(
                decision="reject",
                comment="do not submit yet",
                gated_tool=tool,
            )

        result = maybe_approve(
            store,
            session_id=session.id,
            tool_name="submit",
            prompt=reject,
        )
        assert result is not None
        assert result.decision == "reject"
        feedback = store.list_feedback(session.id)
        assert len(feedback) == 1
        assert feedback[0].comment == "do not submit yet"
    finally:
        store.close()


def test_offline_loop_runs_multiple_tools(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "loop")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _echo_registry()
    try:
        session = store.create_session("improve score", metadata={"max_steps": 5})
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=5,
            auto_approve=True,
        )
        assert len(decisions) >= 2
        tools_run = [d.tool_name for d in decisions if d.tool_name and not d.stop]
        assert "analyze_competition" in tools_run
        # Reopen
        store.close()
        store = ConductorStore(ws.knowledge_dir, ws.competition)
        assert len(store.list_decisions(session.id)) >= 2
        assert len(store.list_tasks(session.id)) >= 2
    finally:
        store.close()


def test_engine_packages_do_not_import_conductor() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    forbidden = "labpilot.research_engine.conductor"
    package_dirs = (
        "intelligence",
        "planner",
        "execution",
        "evidence",
        "reflection",
    )
    violations: list[str] = []
    for name in package_dirs:
        for path in (root / name).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    if module == forbidden or module.startswith(forbidden + "."):
                        violations.append(f"{path}:{node.lineno}")
    assert not violations, "engine must not import conductor:\n" + "\n".join(violations)


def test_search_papers_offline(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "papers")
    result = search_papers(ws, query="audio", offline=True)
    assert result.data["source"] == "offline"
    assert Path(result.refs[0].path or "").is_file()
