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


def test_llm_fallback_allow_deny_retry() -> None:
    from labpilot.research_engine.conductor.policy import llm_next_action

    allow = {"analyze_competition", "search_papers"}
    observe = {"completed_tools": [], "operator_feedback": []}

    class Boom:
        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("boom")

    class Flaky:
        def __init__(self) -> None:
            self.n = 0

        def complete(self, system: str, user: str) -> str:
            self.n += 1
            if self.n < 2:
                raise RuntimeError("transient")
            return (
                '{"tool": "analyze_competition", "args": {}, '
                '"rationale": "ok", "stop": false}'
            )

    allowed = llm_next_action(
        observe,
        allow,
        Boom(),
        offline_fallback_prompt=lambda _r: "allow",
    )
    assert allowed.tool == "analyze_competition"
    assert "offline policy" in allowed.rationale

    denied = llm_next_action(
        observe,
        allow,
        Boom(),
        offline_fallback_prompt=lambda _r: "deny",
    )
    assert denied.stop and denied.tool is None
    assert "denied" in denied.rationale

    decisions = iter(["retry", "allow"])
    retried_then_allow = llm_next_action(
        observe,
        allow,
        Boom(),
        offline_fallback_prompt=lambda _r: next(decisions),
    )
    assert retried_then_allow.tool == "analyze_competition"

    flaky = Flaky()
    recovered = llm_next_action(
        observe,
        allow,
        flaky,
        offline_fallback_prompt=lambda _r: "retry",
    )
    assert recovered.tool == "analyze_competition"
    assert flaky.n == 2


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
            prefer_offline=True,
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


def test_build_observe_bundle_includes_context_online(tmp_path: Path) -> None:
    from labpilot.research_engine.conductor.policy import build_observe_bundle

    ws = _ws(tmp_path, "ctxobs")
    reports = ws.research_paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "note.md").write_text(
        "mixup helps minority classes on audio", encoding="utf-8"
    )
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("use mixup for imbalance")
        observe = build_observe_bundle(
            store, ws, session.id, include_context=True
        )
        assert "context_summary" in observe
        assert isinstance(observe["context_refs"], list)
        blob = (observe.get("context_summary") or "") + str(observe.get("context_refs"))
        assert "mixup" in blob.lower()
    finally:
        store.close()


def test_build_observe_bundle_skips_context_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    import labpilot.research_engine.context as ctx_mod
    from labpilot.research_engine.conductor import policy as policy_mod

    calls: list[str] = []

    def boom(*_a: object, **_k: object) -> object:
        calls.append("build_context")
        raise AssertionError("build_context must not be called")

    monkeypatch.setattr(ctx_mod, "build_context", boom)

    ws = _ws(tmp_path, "ctxskip")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("goal")
        observe = policy_mod.build_observe_bundle(
            store, ws, session.id, include_context=False
        )
        assert "context_summary" not in observe
        assert "context_refs" not in observe
        assert calls == []
    finally:
        store.close()


def test_decide_next_prefer_offline_does_not_require_context(
    tmp_path: Path, monkeypatch
) -> None:
    import labpilot.research_engine.context as ctx_mod
    from labpilot.research_engine.conductor.policy import decide_next

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("context must not run offline")

    monkeypatch.setattr(ctx_mod, "build_context", boom)

    ws = _ws(tmp_path, "ctxoff")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _echo_registry()
    try:
        session = store.create_session("goal")
        action, observe = decide_next(
            store,
            ws,
            session.id,
            reg,
            prefer_offline=True,
        )
        assert action.tool == "analyze_competition"
        assert "context_summary" not in observe
    finally:
        store.close()


def test_observe_survives_build_context_failure(
    tmp_path: Path, monkeypatch
) -> None:
    import labpilot.research_engine.context as ctx_mod
    from labpilot.research_engine.conductor.policy import build_observe_bundle

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("retrieve down")

    monkeypatch.setattr(ctx_mod, "build_context", boom)

    ws = _ws(tmp_path, "ctxfail")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("still decide")
        observe = build_observe_bundle(
            store, ws, session.id, include_context=True
        )
        assert observe["goal"] == "still decide"
        assert observe["context_summary"] == ""
        assert observe["context_refs"] == []
        assert any("retrieve down" in e for e in observe["context_provider_errors"])
    finally:
        store.close()


def test_llm_policy_prompt_sees_ranked_evidence() -> None:
    from labpilot.research_engine.conductor.policy import _invoke_llm_next_action

    captured: dict[str, str] = {}

    class FakeLLM:
        def complete(self, system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return (
                '{"tool": "analyze_competition", "args": {}, '
                '"rationale": "use mixup evidence", "stop": false}'
            )

    observe = {
        "completed_tools": [],
        "operator_feedback": [],
        "context_summary": "[workspace/note] mixup helps minority classes",
        "context_refs": [
            {
                "id": "workspace:note:1",
                "source": "workspace",
                "kind": "note",
                "score": 0.91,
                "reason": "bm25=2.1 | rank=rel=1.000",
            }
        ],
    }
    action = _invoke_llm_next_action(
        observe, {"analyze_competition", "search_papers"}, FakeLLM()
    )
    assert action.tool == "analyze_competition"
    assert "context_refs" in captured["user"]
    assert "0.91" in captured["user"]
    assert "mixup" in captured["user"].lower()
    assert "context_summary" in captured["system"] or "context_refs" in captured["system"]


# --- @latest id resolution --------------------------------------------------


def test_run_experiment_no_longer_forces_dry_run():
    """A campaign that always dry-runs can never produce a real submission."""
    from labpilot.research_engine.conductor.actions import _default_args

    for tool in ("run_plan", "run_experiment"):
        assert "dry_run" not in _default_args(tool)


def test_step_args_resolve_to_latest_ids():
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "run_plan",
        {"plan_id": "@latest"},
        latest_plan_id="P-007",
        latest_execution_id="E-009",
    )
    assert resolved["plan_id"] == "P-007"

    resolved = resolve_step_args(
        "submit",
        {"execution_id": "@latest"},
        latest_plan_id="P-007",
        latest_execution_id="E-009",
    )
    assert resolved["execution_id"] == "E-009"


def test_step_args_fall_back_to_first_id_on_empty_workspace():
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "run_plan",
        {"plan_id": "@latest", "execution_id": "@latest"},
        latest_plan_id=None,
        latest_execution_id=None,
    )
    assert resolved["plan_id"] == "P-001"
    assert resolved["execution_id"] == "E-001"


def test_step_args_leave_explicit_ids_untouched():
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "run_plan",
        {"plan_id": "P-003"},
        latest_plan_id="P-099",
        latest_execution_id=None,
    )
    assert resolved["plan_id"] == "P-003"


def test_generate_plan_switches_to_hypothesis_once_baseline_exists():
    """Baseline compilation is idempotent — a campaign must iterate elsewhere."""
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "generate_plan",
        {"baseline": True},
        latest_plan_id="P-001",
        latest_execution_id="E-001",
        next_hypothesis_id="H-007",
        baseline_plan_exists=True,
    )
    assert resolved == {"hypothesis_id": "H-007"}


def test_generate_plan_keeps_baseline_when_none_exists_yet():
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "generate_plan",
        {"baseline": True},
        latest_plan_id=None,
        latest_execution_id=None,
        next_hypothesis_id="H-007",
        baseline_plan_exists=False,
    )
    assert resolved == {"baseline": True}


def test_generate_plan_keeps_baseline_when_no_hypothesis_available():
    """Without a hypothesis there is nothing better to ask for."""
    from labpilot.research_engine.conductor.actions import resolve_step_args

    resolved = resolve_step_args(
        "generate_plan",
        {"baseline": True},
        latest_plan_id="P-001",
        latest_execution_id="E-001",
        next_hypothesis_id=None,
        baseline_plan_exists=True,
    )
    assert resolved == {"baseline": True}


def test_conductor_analyze_gathers_kaggle_domain_knowledge():
    """No kernels/discussions => no concepts => no hypotheses => no iteration."""
    from labpilot.research_engine.conductor.actions import _default_args

    args = _default_args("analyze_competition")
    assert args["fetch_kaggle"] is True
    # No analyzer is excluded: papers and repositories feed techniques and
    # beliefs just as kernels do.
    assert "exclude" not in args


# --- precondition-aware tool selection ---------------------------------------


class _FakeWorkspace:
    knowledge_dir = None
    competition = "demo"
    effective_runs_dir = None


def _available(monkeypatch, *, has_plan, has_execution):
    import labpilot.research_engine.conductor.loop as loop_mod
    from labpilot.research_engine.conductor.policy import available_tools

    monkeypatch.setattr(loop_mod, "_latest_plan_id", lambda ws: "P-001" if has_plan else None)
    monkeypatch.setattr(
        loop_mod, "_latest_execution_id", lambda ws: "E-001" if has_execution else None
    )
    catalog = {
        "analyze_competition", "search_papers", "query_memory", "generate_plan",
        "implement", "run_plan", "run_experiment", "reflect", "submit", "submit_learn",
    }
    return available_tools(_FakeWorkspace(), catalog)


def test_fresh_workspace_cannot_reflect_run_or_submit(monkeypatch):
    """Step 1 previously chose `reflect` with nothing to reflect on."""
    tools = _available(monkeypatch, has_plan=False, has_execution=False)
    assert "reflect" not in tools
    assert "run_plan" not in tools
    assert "run_experiment" not in tools
    assert "submit" not in tools
    # Evidence gathering and planning are always legitimate first moves.
    assert {"analyze_competition", "generate_plan", "query_memory"} <= tools


def test_plan_unlocks_running_but_not_submitting(monkeypatch):
    tools = _available(monkeypatch, has_plan=True, has_execution=False)
    assert "run_plan" in tools
    assert "run_experiment" in tools
    assert "reflect" not in tools
    assert "submit" not in tools


def test_execution_unlocks_reflect_and_submit(monkeypatch):
    tools = _available(monkeypatch, has_plan=True, has_execution=True)
    assert {"reflect", "submit", "submit_learn"} <= tools
