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
    compress_candidates,
    rank_candidates,
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
    assert any("BM25" in n or "retrieve" in n for n in bundle.notes)


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


def test_sql_graph_port_neighbors_walks_edges(tmp_path: Path) -> None:
    from labpilot.accessor.sqlite import SqliteClient
    from labpilot.research_engine.intelligence.paths import ResearchPaths

    paths = ResearchPaths(tmp_path, "gdemo").ensure()
    client = SqliteClient(paths.db_path)
    try:
        client.conn.execute(
            "INSERT INTO techniques (id, name, category, summary, confidence, created_at, updated_at) "
            "VALUES ('tech_mixup', 'Mixup', '', '', 0.5, '2020-01-01', '2020-01-01')"
        )
        client.conn.execute(
            "INSERT INTO research_artifacts ("
            "id, type, source, title, summary, confidence, competition_slug, "
            "metadata, techniques, models, datasets, claims, refs, created_at, updated_at"
            ") VALUES ("
            "'exp:1', 'experiment', 'test', 't', 's', 0.5, 'gdemo', "
            "'{}', '[]', '[]', '[]', '[]', '[]', '2020-01-01', '2020-01-01')"
        )
        client.conn.execute(
            "INSERT INTO artifact_techniques (artifact_id, technique_id, relation, weight) "
            "VALUES ('exp:1', 'tech_mixup', 'supports', 1.0)"
        )
        client.conn.execute(
            "INSERT INTO evidence_links ("
            "artifact_id, target_kind, target_id, relation, weight, metadata, created_at"
            ") VALUES ('exp:1', 'evidence_card', 'card-9', 'produced', 1.0, '{}', '2020-01-01')"
        )
        client.conn.commit()
    finally:
        client.close()

    port = SqlGraphPort(knowledge_dir=tmp_path, competition="gdemo")
    tech_neighbors = port.neighbors("tech_mixup")
    assert "exp:1" in tech_neighbors
    art_neighbors = port.neighbors("exp:1")
    assert "tech_mixup" in art_neighbors
    assert "card-9" in art_neighbors
    snap = port.metrics_snapshot()
    assert snap.neighbor_calls == 2
    assert snap.neighbor_nodes_returned >= 2


def test_build_context_includes_graph_metrics() -> None:
    bundle = build_context(ContextRequest(competition="demo", goal="g"))
    assert bundle.graph_metrics.neighbor_calls == 0
    assert any("graph neighbors=" in n for n in bundle.notes)
    assert bundle.bm25_metrics.query_empty or bundle.bm25_metrics.candidates_in == 0


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


def test_retrieve_candidates_bm25_scores_all() -> None:
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
    got, metrics = retrieve_candidates(items, request)
    assert len(got) == 3  # retrieve no longer truncates; compress owns budget
    assert got[0].id == "2"
    assert "bm25=" in got[0].reason
    assert metrics.bm25_applied
    assert metrics.scores_positive >= 1
    assert metrics.top_score > 0


def test_rank_boosts_graph_neighbor() -> None:
    class FakeGraph:
        def neighbors(
            self,
            node_id: str,
            *,
            edge_types: list[str] | None = None,
            limit: int = 20,
            hop_depth: int = 1,
        ) -> list[str]:
            _ = (edge_types, limit, hop_depth)
            if node_id == "seed-a":
                return ["node-b"]
            return []

        def metrics_snapshot(self):
            from labpilot.research_engine.context.graph_metrics import GraphQueryMetrics

            return GraphQueryMetrics(neighbor_calls=1)

    # Only top-3 by score are graph seeds; b sits outside the seed set but is a neighbor.
    items = [
        ContextItem(
            id="a",
            source="t",
            kind="note",
            text="mixup seed",
            score=10.0,
            metadata={"node_id": "seed-a", "competition": "demo"},
        ),
        ContextItem(
            id="c",
            source="t",
            kind="note",
            text="high bm25 no graph",
            score=9.0,
            metadata={"node_id": "lonely", "competition": "demo"},
        ),
        ContextItem(
            id="d",
            source="t",
            kind="note",
            text="filler",
            score=8.0,
            metadata={"node_id": "filler-d", "competition": "demo"},
        ),
        ContextItem(
            id="e",
            source="t",
            kind="note",
            text="filler2",
            score=7.0,
            metadata={"node_id": "filler-e", "competition": "demo"},
        ),
        ContextItem(
            id="b",
            source="t",
            kind="note",
            text="related via graph",
            score=1.0,
            metadata={"node_id": "node-b", "competition": "demo"},
        ),
    ]
    request = ContextRequest(competition="demo", query="mixup")
    ranked = rank_candidates(items, request, graph=FakeGraph())
    by_id = {i.id: i for i in ranked}
    assert by_id["a"].metadata["rank_graph_distance"] == 0
    assert by_id["b"].metadata["rank_graph_distance"] == 1
    assert by_id["e"].metadata["rank_graph"] == 0.0
    assert by_id["b"].metadata["rank_graph"] > by_id["e"].metadata["rank_graph"]
    assert "rank=" in by_id["a"].reason


def test_compress_respects_item_and_char_budget() -> None:
    items = [
        ContextItem(id="1", source="t", kind="note", text="a" * 100, score=1.0),
        ContextItem(id="2", source="t", kind="note", text="b" * 100, score=0.9),
        ContextItem(id="3", source="t", kind="note", text="c" * 100, score=0.8),
    ]
    request = ContextRequest(
        competition="demo",
        max_items=2,
        max_chars=80,
        max_item_chars=50,
    )
    kept = compress_candidates(items, request)
    assert len(kept) <= 2
    assert sum(len(i.text) for i in kept) <= 80
    assert all(len(i.text) <= 50 for i in kept)
    assert kept[0].id == "1"


def test_bundle_serializable_json() -> None:
    bundle = build_context(ContextRequest(competition="demo", goal="g", query="q"))
    raw = bundle.to_json()
    assert '"competition":"demo"' in raw.replace(" ", "")
    restored = ContextBundle.model_validate_json(raw)
    assert restored.request.competition == "demo"


def test_rank_expand_records_graph_metrics() -> None:
    class CountingGraph:
        def __init__(self) -> None:
            from labpilot.research_engine.context.graph_metrics import GraphMetricsCollector

            self.metrics = GraphMetricsCollector()

        def neighbors(
            self,
            node_id: str,
            *,
            edge_types: list[str] | None = None,
            limit: int = 20,
            hop_depth: int = 1,
        ) -> list[str]:
            from labpilot.research_engine.context.graph_metrics import timed_neighbor

            _ = (edge_types, limit)
            with timed_neighbor(self.metrics, hop_depth=hop_depth) as timer:
                timer.result_count = 0
                _ = node_id
                return []

        def metrics_snapshot(self):
            return self.metrics.copy()

    class OneProvider:
        name = "one"

        async def fetch(self, request: ContextRequest) -> list[ContextItem]:
            return [
                ContextItem(
                    id="x",
                    source=self.name,
                    kind="note",
                    text=f"mixup {request.query}",
                    score=1.0,
                    metadata={"competition": request.competition, "node_id": "n1"},
                )
            ]

    graph = CountingGraph()
    bundle = build_context(
        ContextRequest(competition="demo", query="mixup"),
        providers=[OneProvider()],
        graph=graph,
    )
    assert bundle.graph_metrics.neighbor_calls >= 1
    assert any("rank(rel+rec+graph)" in n for n in bundle.notes)


def test_bm25_metrics_no_positive_match() -> None:
    items = [
        ContextItem(
            id="1",
            source="t",
            kind="note",
            text="completely unrelated cooking",
            metadata={"competition": "demo"},
        )
    ]
    request = ContextRequest(competition="demo", query="mixup augmentation")
    got, metrics = retrieve_candidates(items, request)
    assert len(got) == 1
    assert metrics.no_positive_match
    assert metrics.scores_zero == 1


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
