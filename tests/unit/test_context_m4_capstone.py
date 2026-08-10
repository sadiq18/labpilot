"""M4 capstone — Context Engine + Conductor online smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.artifacts.base import ArtifactRef
from labpilot.research_engine.conductor.loop import run_until_stop
from labpilot.research_engine.conductor.policy import build_observe_bundle, decide_next
from labpilot.research_engine.conductor.store import ConductorStore
from labpilot.research_engine.context import ContextRequest, build_context
from labpilot.research_engine.tools.descriptors import ToolDescriptor, ToolResult
from labpilot.research_engine.tools.registry import ToolRegistry
from labpilot.research_engine.workspace_facade import Workspace
from labpilot.workspace import scaffold_workspace


def _ws(tmp_path: Path, slug: str = "m4cap") -> Workspace:
    client = scaffold_workspace(tmp_path / slug, slug)
    ws = Workspace.from_client(client).ensure_roots()
    reports = ws.research_paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "note.md").write_text(
        "mixup helps minority classes on audio competitions",
        encoding="utf-8",
    )
    return ws


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    def echo(workspace: Workspace, **kwargs: object) -> ToolResult:
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

    for name in (
        "analyze_competition",
        "search_papers",
        "query_memory",
        "generate_plan",
        "run_plan",
        "reflect",
    ):
        reg.register(ToolDescriptor(name=name, handler=echo, capability_status="fixed"))
    return reg


class _CapturingLLM:
    """Fake LLM that records the policy user payload then stops after one tool."""

    def __init__(self) -> None:
        self.users: list[str] = []
        self.systems: list[str] = []
        self.n = 0

    def complete(self, system: str, user: str) -> str:
        self.systems.append(system)
        self.users.append(user)
        self.n += 1
        if self.n == 1:
            return (
                '{"tool": "analyze_competition", "args": {}, '
                '"rationale": "use mixup evidence", "stop": false}'
            )
        return '{"tool": null, "args": {}, "rationale": "done", "stop": true}'


def test_context_engine_bundle_offline_fixture(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    bundle = build_context(
        ContextRequest(
            competition=ws.competition,
            goal="use mixup for imbalance",
            query="mixup minority",
            knowledge_dir=ws.knowledge_dir,
            max_items=8,
        )
    )
    assert bundle.items
    blob = bundle.summary().lower() + bundle.to_json().lower()
    assert "mixup" in blob
    assert bundle.bm25_metrics.candidates_in >= 1
    restored = type(bundle).model_validate_json(bundle.to_json())
    assert restored.request.competition == ws.competition


def test_online_observe_attaches_ranked_context(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "m4obs")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("use mixup for imbalance")
        observe = build_observe_bundle(store, ws, session.id, include_context=True)
        assert "context_summary" in observe
        assert isinstance(observe["context_refs"], list)
        blob = (observe["context_summary"] or "") + json.dumps(observe["context_refs"])
        assert "mixup" in blob.lower()
    finally:
        store.close()


def test_online_campaign_llm_sees_context_bundle(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "m4camp")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    llm = _CapturingLLM()
    try:
        session = store.create_session(
            "use mixup for imbalance",
            metadata={"autonomy": 1},
        )
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            _registry(),
            llm_client=llm,
            max_steps=3,
            auto_approve=True,
            prefer_offline=False,
            autonomy=1,
            campaign_mode=True,
        )
        assert llm.users, "online campaign should invoke LLM policy"
        user = llm.users[0]
        payload = json.loads(user)
        observe = payload["observe"]
        assert "context_summary" in observe
        assert "context_refs" in observe
        assert "mixup" in json.dumps(observe).lower()
        assert any(s and "context_refs" in s for s in llm.systems), (
            "system prompt should mention ranked evidence"
        )
        tools = [d.tool_name for d in decisions if d.tool_name]
        assert "analyze_competition" in tools
    finally:
        store.close()


def test_offline_campaign_skips_context_engine(tmp_path: Path, monkeypatch) -> None:
    import labpilot.research_engine.context as ctx_mod

    calls: list[str] = []

    def boom(*_a: object, **_k: object) -> object:
        calls.append("build_context")
        raise AssertionError("offline must not force Context Engine")

    monkeypatch.setattr(ctx_mod, "build_context", boom)

    ws = _ws(tmp_path, "m4off")
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    try:
        session = store.create_session("offline goal", metadata={"autonomy": 1})
        decisions = run_until_stop(
            store,
            ws,
            session.id,
            _registry(),
            llm_client=None,
            max_steps=2,
            auto_approve=True,
            prefer_offline=True,
            autonomy=1,
            campaign_mode=True,
        )
        assert calls == []
        assert any(d.tool_name == "analyze_competition" for d in decisions)
    finally:
        store.close()


def test_decide_next_online_survives_empty_knowledge(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "empty", "empty")
    ws = Workspace.from_client(client).ensure_roots()
    store = ConductorStore(ws.knowledge_dir, ws.competition)
    llm = _CapturingLLM()
    try:
        session = store.create_session("anything")
        action, observe = decide_next(
            store,
            ws,
            session.id,
            _registry(),
            llm_client=llm,
            prefer_offline=False,
            auto_offline_fallback=True,
        )
        assert "context_summary" in observe
        assert "context_refs" in observe
        assert action.tool in {None, "analyze_competition"} or action.stop
    finally:
        store.close()
