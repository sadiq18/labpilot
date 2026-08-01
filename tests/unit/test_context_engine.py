"""Unit tests for the Context Engine skeleton."""

from __future__ import annotations

import ast
from pathlib import Path

import anyio

from labpilot.research_engine.context import (
    ContextBundle,
    ContextItem,
    ContextRequest,
    RIRetrievalProvider,
    SqlGraphPort,
    build_context,
    build_context_async,
)
from labpilot.research_engine.context.providers.ri import research_context_to_items
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext
from labpilot.workspace import scaffold_workspace


def test_sync_build_context_without_knowledge_dir() -> None:
    request = ContextRequest(competition="demo", goal="win", query="baseline")
    bundle = build_context(request)
    assert isinstance(bundle, ContextBundle)
    assert bundle.request.competition == "demo"
    assert bundle.items == []
    assert any("identity assemble" in n for n in bundle.notes)


def test_async_gather_isolates_provider_failure() -> None:
    class OkProvider:
        name = "ok"

        async def fetch(self, request: ContextRequest) -> list[ContextItem]:
            return [
                ContextItem(
                    id="1",
                    source=self.name,
                    kind="note",
                    text=f"goal={request.goal}",
                    score=1.0,
                )
            ]

    class BoomProvider:
        name = "boom"

        async def fetch(self, request: ContextRequest) -> list[ContextItem]:
            raise RuntimeError("provider down")

    async def _main() -> ContextBundle:
        request = ContextRequest(competition="x", goal="g")
        return await build_context_async(
            request, providers=[BoomProvider(), OkProvider()]
        )

    bundle = anyio.run(_main)
    assert len(bundle.items) == 1
    assert bundle.items[0].text == "goal=g"
    assert any("boom" in e for e in bundle.provider_errors)


def test_ri_provider_smoke(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "ws", "ctxdemo")
    knowledge_dir = client.knowledge_dir
    request = ContextRequest(
        competition="ctxdemo",
        goal="find techniques",
        query="augmentation",
        knowledge_dir=knowledge_dir,
        max_items=10,
    )
    bundle = build_context(request, providers=[RIRetrievalProvider()])
    assert isinstance(bundle, ContextBundle)
    assert bundle.provider_errors == []
    # Empty knowledge DB still yields a valid bundle (possibly empty items).
    assert bundle.request.competition == "ctxdemo"


def test_research_context_to_items_flattens_brief() -> None:
    ctx = ResearchContext(
        brief="Use Mixup for minority classes",
        techniques=[{"id": "T1", "name": "Mixup", "confidence": 0.8}],
        papers=[{"label": "Mixup paper", "summary": "mix samples", "score": 0.7}],
    )
    items = research_context_to_items(ctx)
    kinds = {i.kind for i in items}
    assert "brief" in kinds
    assert "technique" in kinds
    assert "paper" in kinds


def test_sql_graph_port_neighbors_records_metrics() -> None:
    port = SqlGraphPort(knowledge_dir=Path("."), competition="demo")
    assert port.neighbors("node-1") == []
    snap = port.metrics_snapshot()
    assert snap.neighbor_calls == 1
    assert snap.neighbor_empty_results == 1


def test_build_context_includes_graph_metrics() -> None:
    bundle = build_context(ContextRequest(competition="demo", goal="g"))
    assert bundle.graph_metrics.neighbor_calls == 0
    assert any("graph_neighbors=" in n for n in bundle.notes)


def test_intelligence_does_not_import_context() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    forbidden = "labpilot.research_engine.context"
    violations: list[str] = []
    for path in (root / "intelligence").rglob("*.py"):
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
    assert not violations, "intelligence must not import context:\n" + "\n".join(
        violations
    )
