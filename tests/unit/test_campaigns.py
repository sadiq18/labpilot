"""Unit and integration tests for M3 Campaign Engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.actions import (
    ResearchAction,
    map_research_action,
    offline_next_research_action,
)
from labpilot.research_engine.conductor.approvals import (
    gated_tools_for_autonomy,
    maybe_approve,
)
from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    evaluate_stops,
)
from labpilot.research_engine.conductor.checkpoint import (
    latest_active_session,
    persist_budgets,
    save_checkpoint,
)
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.metrics import ensure_metrics, record_suggestion
from labpilot.research_engine.conductor.models import ApprovalResult
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.handlers.papers import search_papers
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "camp") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _echo(workspace: Workspace, **kwargs: object) -> ToolResult:
    path = workspace.artifacts_dir / "echo.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(kwargs), encoding="utf-8")
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
        data=dict(kwargs),
    )


def _campaign_registry(*, with_submit: bool = True) -> ToolRegistry:
    reg = ToolRegistry()

    def papers(workspace: Workspace, **kwargs: object) -> ToolResult:
        return search_papers(workspace, offline=True, query="demo")

    for name in (
        "analyze_competition",
        "query_memory",
        "generate_plan",
        "run_plan",
        "reflect",
    ):
        reg.register(ToolDescriptor(name=name, handler=_echo, capability_status="fixed"))
    reg.register(ToolDescriptor(name="search_papers", handler=papers, capability_status="fixed"))
    if with_submit:
        reg.register(ToolDescriptor(name="submit", handler=_echo, capability_status="fixed"))
        reg.register(ToolDescriptor(name="submit_learn", handler=_echo, capability_status="fixed"))
    return reg


# -- plan 2: budgets -------------------------------------------------------


def test_budget_stops_submission_wall_cost_target_plateau() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        evaluate_stops(
            BudgetConfig(max_submissions=2),
            BudgetState(submissions=2),
            now=now,
        )
        == "submission_budget"
    )
    assert (
        evaluate_stops(
            BudgetConfig(max_wall_s=10),
            BudgetState(wall_started_at=(now - timedelta(seconds=11)).isoformat()),
            now=now,
        )
        == "wall_time"
    )
    assert (
        evaluate_stops(
            BudgetConfig(max_cost_usd=1.0),
            BudgetState(llm_cost_usd=1.5),
            now=now,
        )
        == "cost_budget"
    )
    assert (
        evaluate_stops(
            BudgetConfig(target_metric="lb", target_value=0.9, maximize=True),
            BudgetState(last_metric=0.91),
            now=now,
        )
        == "metric_target"
    )
    assert (
        evaluate_stops(
            BudgetConfig(plateau_window=3, plateau_epsilon=0.01),
            BudgetState(metric_history=[0.5, 0.501, 0.502]),
            now=now,
        )
        == "plateau"
    )
    assert evaluate_stops(BudgetConfig(), BudgetState(), now=now) == "none"


def test_loop_stops_on_submission_budget(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "budget")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _campaign_registry()
    try:
        session = store.create_session(
            "win",
            metadata={
                # A *spent* budget, not a zero one. `max_submissions=0` means
                # "never submit" and no longer stops a campaign that has not
                # submitted — see test_no_submit_budget_still_runs.
                "budgets": {"max_submissions": 1},
                "budget_state": {"submissions": 1},
            },
        )
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=5,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        assert any(d.stop and "submission_budget" in (d.rationale or "") for d in decisions)
        assert store.get_session(session.id).status == "completed"
    finally:
        store.close()


# -- plan 3: autonomy ------------------------------------------------------


def test_autonomy_gate_matrix() -> None:
    assert "generate_plan" in gated_tools_for_autonomy(0)
    assert "submit" in gated_tools_for_autonomy(0)
    assert "generate_plan" not in gated_tools_for_autonomy(1)
    assert "submit" in gated_tools_for_autonomy(1)
    assert "submit_learn" in gated_tools_for_autonomy(1)


def test_submit_always_gated_autonomy_1(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "auto")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")
        plan_gate = maybe_approve(
            store,
            session_id=session.id,
            tool_name="generate_plan",
            auto=True,
            autonomy=1,
        )
        assert plan_gate is None
        submit_gate = maybe_approve(
            store,
            session_id=session.id,
            tool_name="submit",
            auto=True,
            autonomy=1,
        )
        assert submit_gate is not None
        assert submit_gate.decision == "approve"
    finally:
        store.close()


# -- plan 4: actions -------------------------------------------------------


def test_map_research_action_compose_and_unmapped() -> None:
    allow = {
        "search_papers",
        "generate_plan",
        "run_plan",
        "reflect",
        "analyze_competition",
    }
    plan = map_research_action(
        ResearchAction(
            intent="Investigate whether augmentation helps minority classes",
            suggested_tools=["search_papers", "generate_plan", "run_plan", "reflect"],
        ),
        allow,
    )
    assert not plan.unmapped
    assert [s.tool for s in plan.steps] == [
        "search_papers",
        "generate_plan",
        "run_plan",
        "reflect",
    ]
    bad = map_research_action(
        ResearchAction(intent="teleport", suggested_tools=["invent_agent"]),
        allow,
    )
    assert bad.unmapped and bad.steps == []
    assert bad.suggestion and "invent_agent" in bad.suggestion

    offline = offline_next_research_action([], allow)
    assert offline.suggested_tools == ["analyze_competition"]


def test_offline_compose_multi_tool_campaign(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "compose")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _campaign_registry()
    try:
        session = store.create_session("compose goal", metadata={"autonomy": 1})
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=8,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        tools = [d.tool_name for d in decisions if d.tool_name]
        assert "analyze_competition" in tools
        assert "generate_plan" in tools
        assert "run_plan" in tools
        assert "reflect" in tools
        tasks = store.list_tasks(session.id)
        # Multi-tool action should create dependency edges
        dep_tasks = [t for t in tasks if t.dependencies]
        assert dep_tasks
    finally:
        store.close()


# -- plan 1: checkpoint ----------------------------------------------------


def test_checkpoint_pause_resume_latest_session(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "ckpt")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _campaign_registry(with_submit=False)
    try:
        session = store.create_session("ckpt goal")
        run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=1,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        cp = save_checkpoint(store, session.id)
        assert cp["task_count"] >= 1
        store.update_session_status(session.id, "paused")
        assert latest_active_session(store) is not None
        assert latest_active_session(store).id == session.id

        store.update_session_status(session.id, "running")
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=2,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        assert len(decisions) >= 1
        assert len(store.list_tasks(session.id)) >= 2
    finally:
        store.close()


# -- plan 5: metrics -------------------------------------------------------


def test_metrics_and_suggestions(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "metrics")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("m")
        ensure_metrics(store, session.id)
        record_suggestion(store, session.id, "Need tool invent_x")
        m = store.get_metrics(session.id)
        assert m is not None
        assert m.no_capability == 1
        assert len(store.list_suggestions(session.id)) == 1

        maybe_approve(
            store,
            session_id=session.id,
            tool_name="submit",
            prompt=lambda t: ApprovalResult(decision="reject", comment="no", gated_tool=t),
            autonomy=1,
        )
        m2 = store.get_metrics(session.id)
        assert m2.human_interventions >= 1
    finally:
        store.close()


def test_loop_records_unmapped_suggestion(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "unmap")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _campaign_registry(with_submit=False)
    try:
        session = store.create_session("unmap")
        with patch(
            "labpilot.research_engine.conductor.loop.offline_next_research_action",
            side_effect=[
                ResearchAction(
                    intent="teleport to top of LB",
                    suggested_tools=["invent_teleport"],
                ),
                ResearchAction(intent="done", stop=True, rationale="stop"),
            ],
        ):
            decisions = run_until_stop(
                store,
                ws,
                session.id,
                reg,
                llm_client=None,
                max_steps=3,
                auto_approve=True,
                prefer_offline=True,
                autonomy=1,
            )
        assert store.get_metrics(session.id).no_capability >= 1
        assert store.list_suggestions(session.id)
        assert any((d.observe or {}).get("unmapped") for d in decisions if d.observe)
    finally:
        store.close()


# -- plan 6: capstone ------------------------------------------------------


def test_capstone_offline_campaign(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "capstone")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = _campaign_registry()
    try:
        session = store.create_session(
            "Win offline",
            metadata={
                "autonomy": 1,
                "budgets": {"max_submissions": 1},
                "budget_state": {},
            },
        )
        # Multi-step campaign
        run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=3,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        store.update_session_status(session.id, "paused")
        save_checkpoint(store, session.id, extra={"stop_reason": "operator_pause"})
        assert latest_active_session(store).status == "paused"

        store.update_session_status(session.id, "running")
        run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=5,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        assert len(store.list_tasks(session.id)) >= 3
        tools = {t.tool_name for t in store.list_tasks(session.id) if t.status == "completed"}
        assert "analyze_competition" in tools

        # Budget stop path — a budget that has been spent. Zero now means
        # "never submit" and lets the campaign keep running.
        persist_budgets(
            store,
            session.id,
            BudgetConfig(max_submissions=1),
            BudgetState(submissions=1),
        )
        store.update_session_status(session.id, "running")
        stops = run_until_stop(
            store,
            ws,
            session.id,
            reg,
            llm_client=None,
            max_steps=2,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
        )
        assert any("submission_budget" in (d.rationale or "") for d in stops)

        # Unmappable suggestion path
        record_suggestion(
            store,
            session.id,
            "Need capability/tool 'invent_agent'",
            context={"intent": "teleport"},
        )
        assert store.get_metrics(session.id).no_capability >= 1
    finally:
        store.close()
