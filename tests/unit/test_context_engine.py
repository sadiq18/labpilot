"""Unit tests for the Context Engine (skeleton + BM25 retrieve)."""

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
    apply_filters,
    bm25_scores,
    build_context,
    build_context_async,
    retrieve_candidates,
    tokenize,
)
from labpilot.research_engine.context.providers.ri import research_context_to_items
from labpilot.research_engine.context.providers.workspace import WorkspaceProvider
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext
from labpilot.workspace import scaffold_workspace


def test_sync_build_context_without_knowledge_dir() -> None:
    request = ContextRequest(competition="demo", goal="win", query="baseline")
    bundle = build_context(request)
    assert isinstance(bundle, ContextBundle)
    assert bundle.request.competition == "demo"
    assert bundle.items == []
    assert any("BM25" in n for n in bundle.notes)


def test_async_gather_isolates_provider_failure() -> None:
    class OkProvider:
        name = "ok"

        async def fetch(self, request: ContextRequest) -> list[ContextItem]:
            return [
                ContextItem(
                    id="1",
                    source=self.name,
                    kind="note",
                    text=f"goal={request.goal} baseline mixup",
                    score=1.0,
                    metadata={"competition": request.competition},
                )
            ]

    class BoomProvider:
        name = "boom"

        async def fetch(self, request: ContextRequest) -> list[ContextItem]:
            raise RuntimeError("provider down")

    async def _main() -> ContextBundle:
        request = ContextRequest(competition="x", goal="g", query="mixup")
        return await build_context_async(
            request, providers=[BoomProvider(), OkProvider()]
        )

    bundle = anyio.run(_main)
    assert len(bundle.items) == 1
    assert "mixup" in bundle.items[0].text
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


def test_bm25_ranks_relevant_document_higher() -> None:
    texts = [
        "completely unrelated cooking recipe",
        "mixup augmentation for minority classes",
        "random forest hyperparameters",
    ]
    scores = bm25_scores(texts, "mixup augmentation")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert tokenize("MixUp-Aug") == ["mixup", "aug"]


def test_filters_exclude_wrong_competition_and_kind() -> None:
    items = [
        ContextItem(
            id="a",
            source="t",
            kind="paper",
            text="mixup paper",
            metadata={"competition": "demo", "status": "available"},
        ),
        ContextItem(
            id="b",
            source="t",
            kind="paper",
            text="other comp",
            metadata={"competition": "other", "status": "available"},
        ),
        ContextItem(
            id="c",
            source="t",
            kind="note",
            text="mixup note",
            metadata={"competition": "demo", "status": "available"},
        ),
        ContextItem(
            id="d",
            source="t",
            kind="paper",
            text="failed paper",
            metadata={"competition": "demo", "status": "failed"},
        ),
    ]
    request = ContextRequest(
        competition="demo",
        kinds=["paper"],
        statuses=["available"],
    )
    kept = apply_filters(items, request)
    assert [i.id for i in kept] == ["a"]


def test_retrieve_candidates_bm25_and_max_items() -> None:
    items = [
        ContextItem(
            id="1",
            source="t",
            kind="note",
            text="leaderboard submission strategy",
            metadata={"competition": "demo"},
        ),
        ContextItem(
            id="2",
            source="t",
            kind="note",
            text="mixup for class imbalance",
            metadata={"competition": "demo"},
        ),
        ContextItem(
            id="3",
            source="t",
            kind="note",
            text="unrelated gardening tips",
            metadata={"competition": "demo"},
        ),
    ]
    request = ContextRequest(
        competition="demo",
        query="mixup imbalance",
        max_items=2,
    )
    got = retrieve_candidates(items, request)
    assert len(got) == 2
    assert got[0].id == "2"
    assert "bm25=" in got[0].reason


def test_workspace_provider_reads_report(tmp_path: Path) -> None:
    client = scaffold_workspace(tmp_path / "ws", "wprov")
    ws_root = client.knowledge_dir
    from labpilot.research_engine.workspace_facade import Workspace

    ws = Workspace.from_competition(ws_root, "wprov")
    reports = ws.research_paths.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "note.md").write_text(
        "mixup helps minority classes on audio", encoding="utf-8"
    )
    request = ContextRequest(
        competition="wprov",
        knowledge_dir=ws_root,
        query="mixup minority",
        max_items=5,
    )
    bundle = build_context(request, providers=[WorkspaceProvider()])
    assert any("mixup" in i.text.lower() for i in bundle.items)
    assert all(i.source == "workspace" for i in bundle.items)


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
