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
    bad = validate_next_action(NextAction(tool="invent_agent", rationale="nope"), allow)
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
            return '{"tool": "analyze_competition", "args": {}, "rationale": "ok", "stop": false}'

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
    (reports / "note.md").write_text("mixup helps minority classes on audio", encoding="utf-8")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("use mixup for imbalance")
        observe = build_observe_bundle(store, ws, session.id, include_context=True)
        assert "context_summary" in observe
        assert isinstance(observe["context_refs"], list)
        blob = (observe.get("context_summary") or "") + str(observe.get("context_refs"))
        assert "mixup" in blob.lower()
    finally:
        store.close()


def test_build_observe_bundle_skips_context_when_disabled(tmp_path: Path, monkeypatch) -> None:
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
        observe = policy_mod.build_observe_bundle(store, ws, session.id, include_context=False)
        assert "context_summary" not in observe
        assert "context_refs" not in observe
        assert calls == []
    finally:
        store.close()


def test_decide_next_prefer_offline_does_not_require_context(tmp_path: Path, monkeypatch) -> None:
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


def test_observe_survives_build_context_failure(tmp_path: Path, monkeypatch) -> None:
    import labpilot.research_engine.context as ctx_mod
    from labpilot.research_engine.conductor.policy import build_observe_bundle

    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("retrieve down")

    monkeypatch.setattr(ctx_mod, "build_context", boom)

    ws = _ws(tmp_path, "ctxfail")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("still decide")
        observe = build_observe_bundle(store, ws, session.id, include_context=True)
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
    action = _invoke_llm_next_action(observe, {"analyze_competition", "search_papers"}, FakeLLM())
    assert action.tool == "analyze_competition"
    assert "context_refs" in captured["user"]
    assert "0.91" in captured["user"]
    assert "mixup" in captured["user"].lower()
    assert "context_summary" in captured["system"] or "context_refs" in captured["system"]


# --- @latest id resolution --------------------------------------------------


def test_run_experiment_no_longer_forces_dry_run():
    """A campaign that always dry-runs can never produce a real submission.

    Asserting merely that the key is absent is not enough, and previously gave
    false confidence: `run_experiment` defaults dry_run=True in its own
    signature, so omitting it left every campaign run a dry run.
    """
    from labpilot.research_engine.conductor.actions import _default_args

    for tool in ("run_plan", "run_experiment"):
        assert _default_args(tool).get("dry_run") is False


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
    """No kernels => no concepts => no hypotheses => no iteration."""
    from labpilot.research_engine.conductor.actions import _default_args

    args = _default_args("analyze_competition")
    assert args["fetch_kaggle"] is True


def test_a_campaign_buys_the_cheap_slice_of_the_evidence():
    """A campaign runs this every few steps, so it takes the highest-yield
    source and skips the rest.

    The paper analyzer searches 40 and LLM-extracts 15; on rogii 2026-08-09 it
    held a single step for over sixteen minutes without finishing. Kernels
    sorted by score are the evidence that actually ran against this dataset.
    """
    from labpilot.research_engine.conductor.actions import _default_args

    args = _default_args("analyze_competition")
    assert args["kaggle_fetch_plan"] == "best_score"
    assert args["exclude"] == ["papers"]


def test_the_campaign_and_the_template_ask_for_the_same_thing():
    """Two call sites, one budget — they drifted apart once already."""
    from labpilot.research_engine.conductor.actions import (
        _TEMPLATES,
        _default_args,
    )

    template_args = [
        step.args
        for _keywords, steps in _TEMPLATES
        for step in steps
        if step.tool == "analyze_competition"
    ]
    assert template_args
    for args in template_args:
        assert args == _default_args("analyze_competition")


# --- precondition-aware tool selection ---------------------------------------


class _FakeWorkspace:
    knowledge_dir = None
    competition = "demo"
    effective_runs_dir = None


def _available(monkeypatch, *, has_plan, has_execution):
    import labpilot.research_engine.conductor.loop as loop_mod
    import labpilot.research_engine.conductor.policy as policy_mod
    from labpilot.research_engine.conductor.policy import available_tools

    # `run_plan` is gated on a plan the Engineer would actually accept, not on
    # "a plan exists" — patch the predicate that decides it.
    monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws: has_plan)
    monkeypatch.setattr(
        loop_mod, "_latest_execution_id", lambda ws: "E-001" if has_execution else None
    )
    monkeypatch.setattr(policy_mod, "untested_hypothesis_count", lambda ws: 0)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: None)
    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: False)
    catalog = {
        "analyze_competition",
        "search_papers",
        "query_memory",
        "generate_plan",
        "implement",
        "run_plan",
        "run_experiment",
        "reflect",
        "submit",
        "submit_learn",
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


# --- goal persistence --------------------------------------------------------


class _Cfg:
    def __init__(self, metric="mse", value=5.0, maximize=False):
        self.target_metric = metric
        self.target_value = value
        self.maximize = maximize


class _State:
    def __init__(self, last=None):
        self.last_metric = last


def test_objective_unmet_when_target_not_reached():
    from labpilot.research_engine.conductor.loop import _objective_unmet

    # minimising: 194.8 is far above the target of 5
    assert _objective_unmet(_Cfg(), _State(194.8)) is True
    assert _objective_unmet(_Cfg(), _State(4.2)) is False
    # nothing measured yet still counts as unmet
    assert _objective_unmet(_Cfg(), _State(None)) is True


def test_objective_unmet_respects_maximise_direction():
    from labpilot.research_engine.conductor.loop import _objective_unmet

    cfg = _Cfg(metric="accuracy", value=0.9, maximize=True)
    assert _objective_unmet(cfg, _State(0.7)) is True
    assert _objective_unmet(cfg, _State(0.95)) is False


def test_no_target_means_policy_stop_is_honoured():
    """Without an objective there is nothing to persist toward."""
    from labpilot.research_engine.conductor.loop import _objective_unmet

    assert _objective_unmet(_Cfg(metric=None, value=None), _State(1.0)) is False


# --- backlog-aware scheduling -------------------------------------------------


def _available_with_backlog(monkeypatch, backlog, *, has_plan=True, has_execution=True):
    import labpilot.research_engine.conductor.loop as loop_mod
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws: has_plan)
    monkeypatch.setattr(
        loop_mod, "_latest_execution_id", lambda ws: "E-001" if has_execution else None
    )
    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: backlog)
    # Fresh, so the *pool* decides. Staleness is exercised separately —
    # pinning it at 99h here would make every case gather and prove nothing.
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)
    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: False)
    catalog = {
        "analyze_competition",
        "search_papers",
        "query_memory",
        "generate_plan",
        "implement",
        "run_plan",
        "run_experiment",
        "reflect",
        "submit",
    }
    return policy_mod.available_tools(_FakeWorkspace(), catalog)


def test_a_full_pool_with_fresh_evidence_blocks_re_gathering(monkeypatch):
    """Re-analysing with viable work queued and a recent sweep costs minutes and
    adds nothing. Note *viable* and *fresh* — a full pool alone no longer
    blocks, because that was the ratchet."""
    tools = _available_with_backlog(monkeypatch, backlog=10)
    assert "analyze_competition" not in tools
    assert "search_papers" not in tools
    # Testing what is queued stays available.
    assert {"generate_plan", "run_plan", "reflect"} <= tools


def test_an_empty_pool_reopens_evidence_gathering(monkeypatch):
    tools = _available_with_backlog(monkeypatch, backlog=0)
    assert "analyze_competition" in tools


def test_search_papers_is_never_offered_to_a_campaign(monkeypatch):
    """It is forced `offline=True` on every conductor path and the policy only
    picks tool names, so it can write `count: 0` and nothing else. An empty
    pool is exactly when a wasted step hurts most."""
    assert "search_papers" not in _available_with_backlog(monkeypatch, backlog=0)
    assert "search_papers" not in _available_with_backlog(monkeypatch, backlog=10)


def test_backlog_below_target_still_gathers(monkeypatch):
    """A thin queue is worth topping up before it runs dry."""
    tools = _available_with_backlog(monkeypatch, backlog=2)
    assert "analyze_competition" in tools


def test_a_re_sweep_inside_the_floor_is_refused(monkeypatch):
    """Making the clauses independent introduced a failure the AND version
    could not have: a pool that stays thin would sweep every step. The floor is
    a rate limit under both clauses, not a third gate."""
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 0)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 0.1)
    ok, reason = policy_mod.should_gather_evidence(_FakeWorkspace())
    assert ok is False
    assert "minutes ago" in reason


def test_a_thin_pool_reopens_gathering(monkeypatch):
    """Fewer viable ideas than the target is reason enough, however fresh the
    evidence — the pool being empty is the bottleneck."""
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 1)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)
    ok, reason = policy_mod.should_gather_evidence(_FakeWorkspace())
    assert ok is True
    assert "viable" in reason


def test_stale_evidence_reopens_gathering_however_full_the_queue(monkeypatch):
    """The ratchet, inverted deliberately.

    This test previously asserted the opposite — *"a full queue blocks
    gathering however old the evidence is"* — and that assertion is the defect.
    On rogii 2026-08-09 it kept 46 stale hypotheses holding `analyze_competition`
    and `search_papers` out of the allowlist permanently, so the only thing that
    could refresh the pool was disabled by the pool.

    A queue of stale ideas is the strongest reason to find better ones.
    """
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 50)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 999.0)
    ok, reason = policy_mod.should_gather_evidence(_FakeWorkspace())
    assert ok is True
    assert "old" in reason


def test_a_full_fresh_pool_does_not_gather(monkeypatch):
    """The carve-out must not cost the brake: both clauses unmet means testing
    is the better use of the step."""
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 9)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 2.0)
    ok, _ = policy_mod.should_gather_evidence(_FakeWorkspace())
    assert ok is False


def test_never_gathered_always_allows_gathering(monkeypatch):
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "viable_hypothesis_count", lambda kd, c: 9)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: None)
    ok, reason = policy_mod.should_gather_evidence(_FakeWorkspace())
    assert ok is True
    assert "no evidence" in reason


def test_unrun_plan_blocks_queuing_another(monkeypatch):
    """The campaign chose generate_plan three steps running without executing one."""
    import labpilot.research_engine.conductor.loop as loop_mod
    import labpilot.research_engine.conductor.policy as policy_mod

    monkeypatch.setattr(policy_mod, "has_runnable_plan", lambda ws: True)
    monkeypatch.setattr(loop_mod, "_latest_execution_id", lambda ws: "E-001")
    monkeypatch.setattr(policy_mod, "untested_hypothesis_count", lambda ws: 5)
    monkeypatch.setattr(policy_mod, "hours_since_last_artifact", lambda ws: 1.0)
    monkeypatch.setattr(policy_mod, "has_unrun_plan", lambda ws: True)

    tools = policy_mod.available_tools(_FakeWorkspace(), {"generate_plan", "run_plan", "reflect"})
    assert "generate_plan" not in tools
    assert "run_plan" in tools


def test_latest_plan_prefers_a_runnable_one(tmp_path):
    """Against a real store, because the runnable filter now lives in SQL.

    A faked plan source would exercise none of it — the join is where "runnable"
    and "not testing a retired idea" are actually decided.
    """
    from datetime import UTC, datetime

    import labpilot.research_engine.conductor.loop as loop_mod
    from labpilot.research_engine.planner.schemas.models import ResearchPlan
    from labpilot.research_engine.planner.schemas.task_types import PlanStatus
    from labpilot.research_engine.planner.store import PlanStore

    store = PlanStore(tmp_path / "knowledge", "demo")
    try:
        for pid, status in (
            ("P-001", PlanStatus.DONE),
            ("P-002", PlanStatus.READY),
            ("P-008", PlanStatus.DONE),
        ):
            now = datetime.now(UTC)
            store.upsert_plan(
                ResearchPlan(
                    id=pid,
                    competition="demo",
                    hypothesis_id="",
                    goal="g",
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
            )
    finally:
        store.close()

    class _WS:
        knowledge_dir = tmp_path / "knowledge"
        competition = "demo"

    assert loop_mod._latest_plan_id(_WS()) == "P-002"


def test_latest_plan_is_none_when_none_runnable(monkeypatch, tmp_path):
    """No runnable plan means no answer — not "the newest done one".

    This test previously asserted the fallback. It was encoding the bug: the
    Engineer refuses a done plan with "need ready or in_progress", so returning
    one made the Conductor offer `run_plan` and lose a step every time. `None`
    lets it offer `generate_plan` instead.
    """
    import labpilot.research_engine.conductor.loop as loop_mod

    class _Plan:
        def __init__(self, pid):
            self.id = pid
            self.status = "done"
            self.metadata = {}

    class _Artifacts:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return [_Plan("P-001"), _Plan("P-008")]

        def close(self):
            pass

    monkeypatch.setattr("labpilot.research_engine.artifacts.plan.PlanArtifacts", _Artifacts)

    class _WS:
        knowledge_dir = tmp_path
        competition = "demo"

    assert loop_mod._latest_plan_id(_WS()) is None


def test_campaign_runs_are_not_dry_runs():
    """run_experiment defaults dry_run=True in its own signature, so the
    Conductor must say otherwise or it renders code and never trains."""
    from labpilot.research_engine.conductor.actions import _TEMPLATES, _default_args

    for tool in ("run_plan", "run_experiment"):
        assert _default_args(tool)["dry_run"] is False

    for _keywords, steps in _TEMPLATES:
        for step in steps:
            if step.tool in {"run_plan", "run_experiment"}:
                assert step.args.get("dry_run") is False, step.tool


def test_non_dry_experiment_without_metrics_is_a_failure(monkeypatch):
    """Silent no-op protection: 'completed' with no metrics is not success."""
    import pytest

    from labpilot.research_engine.tools.handlers import specialists

    monkeypatch.setattr(specialists, "execute_agent_sync", lambda *a, **k: [])
    monkeypatch.setattr(specialists, "_bundle", lambda *a, **k: None)

    class _Cand:
        name = "experiment"
        agent = object()

    class _Reg:
        def candidates(self, capability):
            return [_Cand()]

    monkeypatch.setattr(specialists, "build_default_specialist_registry", lambda **k: _Reg())

    with pytest.raises(specialists.ExperimentProducedNoMetricsError):
        specialists.run_experiment(object(), plan_id="P-009", dry_run=False)

    # A dry run legitimately produces nothing and must stay allowed.
    result = specialists.run_experiment(object(), plan_id="P-009", dry_run=True)
    assert result.data["dry_run"] is True


def test_goal_persistence_override_dispatches_reflect_not_generate_plan():
    """Regression: routing the override by intent text let keyword matching
    hijack it — the phrase contained "hypothesis", which matches the
    ("plan", "baseline", "hypothesis") template, so it dispatched
    generate_plan(baseline=True) instead of reflecting on the result."""
    from labpilot.research_engine.conductor.actions import (
        ResearchAction,
        map_research_action,
    )

    catalog = {"reflect", "generate_plan", "run_experiment", "analyze_competition"}
    action = ResearchAction(
        intent="objective unmet — reflect and continue",
        rationale="objective still unmet; continuing",
        suggested_tools=["reflect"],
    )
    plan = map_research_action(action, catalog)
    assert [s.tool for s in plan.steps] == ["reflect"]


def test_intent_text_alone_would_still_be_hijacked():
    """Documents *why* the override names its tool explicitly."""
    from labpilot.research_engine.conductor.actions import (
        ResearchAction,
        map_research_action,
    )

    catalog = {"reflect", "generate_plan", "run_experiment"}
    hijacked = map_research_action(
        ResearchAction(intent="reflect on the last experiment and try the next hypothesis"),
        catalog,
    )
    assert [s.tool for s in hijacked.steps] != ["reflect"]


# --- review #96: strict mode must stop the session, not just the step -------


def test_strict_llm_abort_stops_the_campaign(tmp_path, monkeypatch):
    """`LLMDegradedError` was caught by the generic dispatch handler, which
    logged "Task failed" and carried on. A degraded LLM that merely costs a
    step is the silent degradation M14 exists to remove, one level up.
    """
    import pytest as _pytest

    from labpilot.accessor.common.micro_agents import LLMDegradedError
    from labpilot.research_engine.conductor import loop as loop_mod

    calls: list[str] = []

    class _Store:
        def append_decision(self, record):
            calls.append("decision")

        def increment_metric(self, session_id, name):
            calls.append(f"metric:{name}")

        def update_session_status(self, session_id, status):
            calls.append(f"status:{status}")

    class _Record:
        rationale = "chose implement"

    decisions: list = []
    loop_mod._fail_session_on_degraded_llm(
        _Store(), "S-1", _Record(), decisions, LLMDegradedError("agent: prose reply")
    )

    assert "status:failed" in calls, "the session must be marked, not left short"
    assert "metric:tasks_failed" in calls
    assert len(decisions) == 1
    assert "strict LLM abort" in decisions[0].rationale
    _ = _pytest


def test_the_degraded_handler_precedes_the_generic_one():
    """Ordering is the whole fix: a later `except Exception` would swallow it."""
    import inspect

    from labpilot.research_engine.conductor import loop as loop_mod

    source = (
        inspect.getsource(loop_mod.run_until_stop.__wrapped__)
        if hasattr(loop_mod.run_until_stop, "__wrapped__")
        else inspect.getsource(loop_mod._run_until_stop_inner)
    )
    degraded = source.index("except LLMDegradedError")
    generic = source.index("except Exception as exc:", degraded)
    assert degraded < generic, "LLMDegradedError must be caught before Exception"


# --- the offline policy must not fabricate evidence (#40) -------------------


def test_offline_policy_never_requests_a_dry_run():
    """The guard that existed and did not cover this path.

    `test_campaign_runs_are_not_dry_runs` checks `_default_args` and the
    templates. `offline_next_action` hand-rolled its own args with
    `dry_run=True`, so it was never covered. A dry run writes
    `{'cv_accuracy': 0.5, 'status': 'dry_run_stub'}`, which is the source of 6
    of the 7 placeholder evidence cards that fabricated `hyp:H-010`'s -971.50
    net effect on rogii. It fired this way twice in real campaigns.
    """
    from labpilot.research_engine.conductor.policy import _DEFAULT_ORDER, offline_next_action

    for tool in _DEFAULT_ORDER:
        action = offline_next_action({"completed_tools": []}, {tool})
        assert action.tool == tool
        assert action.args.get("dry_run") is not True, f"{tool} would run a stub"


def test_offline_policy_does_not_pin_a_plan_id():
    """`plan_id="P-001"` was a placeholder that may be done or may not exist.

    It existed because the legacy dispatch path did not resolve `@latest`;
    it does now, so the shared sentinel works here as it does everywhere else.
    """
    from labpilot.research_engine.conductor.actions import LATEST
    from labpilot.research_engine.conductor.policy import offline_next_action

    action = offline_next_action({"completed_tools": []}, {"run_plan"})
    assert action.args.get("plan_id") == LATEST


def test_offline_policy_matches_the_shared_defaults():
    """Two sources of args is what let them drift apart in the first place."""
    from labpilot.research_engine.conductor.actions import _default_args
    from labpilot.research_engine.conductor.policy import _DEFAULT_ORDER, offline_next_action

    for tool in _DEFAULT_ORDER:
        action = offline_next_action({"completed_tools": []}, {tool})
        assert action.args == _default_args(tool), tool


def test_an_exhausted_catalog_cycles_instead_of_stopping():
    """`stop=True` here ended two healthy campaigns — S-019 at step 8 and
    S-020 at step 27 — with the LLM working fine by then. A research loop is
    not a one-pass checklist."""
    from labpilot.research_engine.conductor.policy import offline_next_action

    done = [
        "analyze_competition",
        "search_papers",
        "query_memory",
        "generate_plan",
        "run_plan",
        "reflect",
        "submit",
    ]
    action = offline_next_action(
        {"completed_tools": done}, {"generate_plan", "run_plan", "reflect"}
    )
    assert action.stop is False
    assert action.tool in {"generate_plan", "run_plan", "reflect"}


def test_cycling_still_stops_at_a_genuine_dead_end():
    """The allowlist does the anti-spin work: `generate_plan` is gated while a
    plan is unrun and `run_plan` while none is runnable, so an empty allowlist
    is a real dead end rather than a reason to loop."""
    from labpilot.research_engine.conductor.policy import offline_next_action

    action = offline_next_action({"completed_tools": ["generate_plan"]}, set())
    assert action.stop is True
    assert action.tool is None


def test_cycling_honours_operator_rejection():
    from labpilot.research_engine.conductor.policy import offline_next_action

    # `reflect` is deliberately not the last completed tool: with it last, the
    # spin guard fires and this would assert the wrong rule — the same "whatever
    # fires first" trap as the credential test in PR #98.
    observe = {
        "completed_tools": ["reflect", "generate_plan", "run_plan"],
        "operator_feedback": [{"decision": "reject", "gated_tool": "run_plan"}],
    }
    action = offline_next_action(observe, {"run_plan", "reflect"})
    assert action.stop is False
    assert action.tool == "reflect", "rejected run_plan must not be chosen"


def test_cycling_is_least_recently_used_not_fixed_order():
    """Fixed `_REPEATABLE` order let `generate_plan` win whenever available,
    giving plan -> run -> plan -> run and never reflecting again."""
    from labpilot.research_engine.conductor.policy import offline_next_action

    full = {"generate_plan", "run_plan", "reflect"}
    done = [
        "analyze_competition",
        "search_papers",
        "query_memory",
        "generate_plan",
        "run_plan",
        "reflect",
        "submit",
    ]

    seen = []
    for _ in range(3):
        tool = offline_next_action({"completed_tools": done}, full).tool
        seen.append(tool)
        done.append(tool)
    assert seen == ["generate_plan", "run_plan", "reflect"], seen


def test_cycling_stops_rather_than_repeating_itself():
    """A DRAFT plan makes `has_unrun_plan` true and `has_runnable_plan` false,
    so the allowlist can narrow to `{reflect}` — bounded by max_steps, but it
    burns a whole degraded campaign re-reflecting on the same state."""
    from labpilot.research_engine.conductor.policy import offline_next_action

    done = ["generate_plan", "run_plan", "reflect"]
    action = offline_next_action({"completed_tools": done}, {"reflect"})
    assert action.stop is True
    assert "just ran" in action.rationale


def test_a_sole_tool_that_has_not_just_run_is_still_offered():
    """The spin guard must not refuse work that is genuinely next."""
    from labpilot.research_engine.conductor.policy import offline_next_action

    done = ["generate_plan", "run_plan", "reflect", "generate_plan"]
    action = offline_next_action({"completed_tools": done}, {"reflect"})
    assert action.stop is False
    assert action.tool == "reflect"


def test_the_legacy_dispatch_path_resolves_latest(monkeypatch, tmp_path):
    """`@latest` must not reach the Engineer as a literal.

    Only the multi-step campaign path resolved it, which is why
    `offline_next_action` pinned `plan_id="P-001"` instead of using the shared
    defaults. This pins the parity that removed the hardcode.
    """
    from labpilot.research_engine.conductor import loop as loop_mod
    from labpilot.research_engine.conductor.actions import LATEST, resolve_step_args

    monkeypatch.setattr(loop_mod, "_latest_plan_id", lambda ws: "P-042")
    monkeypatch.setattr(loop_mod, "_latest_execution_id", lambda ws: "E-007")
    monkeypatch.setattr(loop_mod, "_next_hypothesis_id", lambda ws: None)
    monkeypatch.setattr(loop_mod, "_baseline_plan_exists", lambda ws: True)

    resolved = resolve_step_args(
        "run_plan",
        {"plan_id": LATEST, "dry_run": False},
        latest_plan_id=loop_mod._latest_plan_id(None),
        latest_execution_id=loop_mod._latest_execution_id(None),
        next_hypothesis_id=None,
        baseline_plan_exists=True,
    )
    assert resolved["plan_id"] == "P-042", "the sentinel reached dispatch unresolved"
    assert resolved["dry_run"] is False


def test_the_loop_resolves_before_enqueue_on_both_paths():
    """Guards the parity itself: if someone removes the legacy resolve, the
    hardcoded-plan-id bug becomes reachable again."""
    import inspect

    from labpilot.research_engine.conductor import loop as loop_mod

    source = inspect.getsource(loop_mod._run_until_stop_inner)
    assert source.count("resolve_step_args(") >= 2, (
        "both the multi-step and legacy dispatch paths must resolve @latest"
    )
