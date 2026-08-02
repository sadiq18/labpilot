"""Unit tests for capability gap ledger, export, and registration helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.gap_ledger import (
    apply_gap_decision,
    build_suggestion_context,
    export_gaps_payload,
    is_maintainer_enabled,
    normalize_gap_key,
    note_suggestion,
    redact_context,
)
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.metrics import record_suggestion
from labpilot.research_engine.conductor.actions import ResearchAction
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registration import register_tool
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace
from unittest.mock import patch


def _ws(tmp_path: Path, slug: str = "gaps") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    return Workspace.from_client(client).ensure_roots()


def _echo(workspace: Workspace, **kwargs: object) -> ToolResult:
    path = workspace.artifacts_dir / "echo.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok", encoding="utf-8")
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
        data={},
    )


def test_normalize_gap_key_prefers_tool_then_intent() -> None:
    assert normalize_gap_key(missing_tools=["Run_EDA"]) == "tool:run_eda"
    assert (
        normalize_gap_key(
            message="Need capability/tool 'invent_x' for intent: teleport"
        )
        == "tool:invent_x"
    )
    assert normalize_gap_key(intent="  Teleport Now  ").startswith("intent:teleport")


def test_redact_context_drops_goal_and_session() -> None:
    raw = build_suggestion_context(
        intent="x",
        missing_tools=["t"],
        competition="c",
        session_id="S-001",
        goal="secret goal",
    )
    red = redact_context(raw)
    assert "goal" not in red
    assert "session_id" not in red
    assert red["missing_tools"] == ["t"]
    assert red["competition"] == "c"


def test_record_suggestion_upserts_gap_ledger(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("g")
        ctx = build_suggestion_context(
            intent="teleport",
            missing_tools=["invent_teleport"],
            competition=ws.competition,
            session_id=session.id,
            goal=session.goal,
        )
        record_suggestion(
            store,
            session.id,
            "Need capability/tool 'invent_teleport' for intent: teleport",
            context=ctx,
        )
        record_suggestion(
            store,
            session.id,
            "Need capability/tool 'invent_teleport' for intent: teleport",
            context=ctx,
        )
        gaps = store.list_capability_gaps()
        assert len(gaps) == 1
        assert gaps[0].gap_key == "tool:invent_teleport"
        assert gaps[0].count == 2
        assert gaps[0].status == "open"
        payload = export_gaps_payload(store, competition=ws.competition)
        assert payload["schema"] == "labpilot.capability_gaps/v1"
        assert payload["gaps"][0]["count"] == 2
        assert "goal" not in payload["gaps"][0]["sample_contexts"][0]
    finally:
        store.close()


def test_apply_gap_decision_requires_maintainer(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "dec")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("d")
        record_suggestion(
            store,
            session.id,
            "Need capability/tool 'x'",
            context={"intent": "i", "missing_tools": ["x"]},
        )
        with pytest.raises(PermissionError):
            apply_gap_decision(
                store,
                "tool:x",
                "promote",
                promoted_tool="x",
                env={},
            )
        assert not is_maintainer_enabled({})
        decision = apply_gap_decision(
            store,
            "tool:x",
            "promote",
            promoted_tool="run_x",
            reason="recurring gap",
            env={"LABPILOT_MAINTAINER": "1"},
        )
        assert decision.decision == "promote"
        gap = store.get_capability_gap("tool:x")
        assert gap is not None
        assert gap.status == "promoted"
        assert gap.promoted_tool == "run_x"
        assert len(store.list_capability_decisions("tool:x")) == 1
    finally:
        store.close()


def test_register_tool_and_allowlist_refresh(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "live")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    reg = ToolRegistry()
    reg.register(ToolDescriptor(name="analyze_competition", handler=_echo))

    calls = {"n": 0}

    def side_effect(completed, allowlist):
        calls["n"] += 1
        if calls["n"] == 1:
            return ResearchAction(
                intent="need new",
                suggested_tools=["brand_new_tool"],
            )
        if calls["n"] == 2:
            register_tool(
                reg,
                ToolDescriptor(name="brand_new_tool", handler=_echo),
            )
            return ResearchAction(
                intent="use new",
                suggested_tools=["brand_new_tool"],
            )
        return ResearchAction(intent="done", stop=True, rationale="stop")

    try:
        session = store.create_session("live-reg")
        with patch(
            "labpilot.research_engine.conductor.loop.offline_next_research_action",
            side_effect=side_effect,
        ):
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
        assert store.get_metrics(session.id).no_capability >= 1
        suggestions = store.list_suggestions(session.id)
        assert suggestions
        assert suggestions[0].context.get("missing_tools") == ["brand_new_tool"]
        completed = {
            t.tool_name for t in store.list_tasks(session.id) if t.status == "completed"
        }
        assert "brand_new_tool" in completed
        assert any(not d.stop for d in decisions)
    finally:
        store.close()


def test_note_suggestion_from_message_only(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "msg")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("m")
        from labpilot.research_engine.conductor.metrics import Suggestion

        s = Suggestion(
            id=store.new_suggestion_id(),
            session_id=session.id,
            message="Need capability/tool 'solo'",
            context={},
        )
        store.append_suggestion(s)
        gap = note_suggestion(store, s)
        assert gap.gap_key == "tool:solo"
    finally:
        store.close()
