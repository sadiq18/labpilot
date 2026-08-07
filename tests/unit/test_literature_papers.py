"""Plan 6 — literature clients, cache, PaperAnalyzer (no live network)."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import httpx
import pytest

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.analyzers.papers import PaperAnalyzer
from labpilot.research_engine.intelligence.literature.cache import PaperCatalogStore
from labpilot.research_engine.intelligence.literature.clients import (
    ArxivClient,
    HuggingFacePapersClient,
    OpenAlexClient,
    SemanticScholarClient,
    normalize_arxiv_id,
    stable_paper_id,
)
from labpilot.research_engine.intelligence.literature.models import Paper, PaperKnowledge
from labpilot.research_engine.intelligence.literature.provider import ChainedLiteratureProvider
from labpilot.research_engine.intelligence.literature.query import build_literature_query
from labpilot.research_engine.intelligence.micro_agents.paper_analyzer import (
    PaperAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import AnalyzeContext, ResearchArtifactType


def _ctx(tmp_path: Path, competition: str = "birdclef-2026") -> AnalyzeContext:
    runs = tmp_path / "runs"
    knowledge = tmp_path / "knowledge"
    runs.mkdir()
    knowledge.mkdir()
    return AnalyzeContext(
        competition=competition,
        runs_dir=runs,
        knowledge_dir=knowledge,
        refresh=False,
    )


def test_normalize_and_stable_ids() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2106.09685v2") == "2106.09685"
    assert stable_paper_id(doi="10.1/abc", arxiv_id=None, s2_id="x") == "doi:10.1/abc"
    assert stable_paper_id(doi=None, arxiv_id="2106.09685", s2_id="x").startswith("arxiv:")


def test_semantic_scholar_normalize_fixture() -> None:
    client = SemanticScholarClient()
    paper = client._normalize(
        {
            "paperId": "abc123",
            "title": "SpecAugment",
            "abstract": "We propose time and frequency masking.",
            "year": 2019,
            "venue": "Interspeech",
            "citationCount": 2000,
            "authors": [{"name": "Park"}],
            "externalIds": {"ArXiv": "1904.08779", "DOI": "10.1/spec"},
            "url": "https://www.semanticscholar.org/paper/abc123",
            "openAccessPdf": {"url": "https://example.com/a.pdf"},
        },
        rank_hint=0.9,
    )
    assert paper is not None
    assert paper.title == "SpecAugment"
    assert paper.arxiv_id == "1904.08779"
    assert paper.doi == "10.1/spec"
    assert paper.citations == 2000
    assert paper.id.startswith("doi:")


def test_openalex_merge_enrichment() -> None:
    client = OpenAlexClient()
    base = Paper(id="arxiv:1", title="T", abstract="a", relevance=0.5)
    merged = client._merge(
        base,
        {
            "cited_by_count": 42,
            "publication_year": 2020,
            "concepts": [{"display_name": "speech"}],
            "open_access": {"oa_url": "https://example.com/oa.pdf"},
            "ids": {"openalex": "https://openalex.org/W1", "doi": "https://doi.org/10.x/y"},
        },
    )
    assert merged.citations == 42
    assert merged.year == 2020
    assert "speech" in merged.concepts
    assert merged.pdf_url.endswith("oa.pdf")
    assert merged.doi == "10.x/y"


def test_openalex_merge_pulls_arxiv_from_primary_location() -> None:
    client = OpenAlexClient()
    base = Paper(id="doi:10.x/y", title="T", abstract="a", doi="10.x/y", relevance=0.5)
    merged = client._merge(
        base,
        {
            "cited_by_count": 10,
            "ids": {"openalex": "https://openalex.org/W2", "doi": "https://doi.org/10.x/y"},
            "primary_location": {
                "landing_page_url": "https://arxiv.org/abs/2106.09685v2",
                "pdf_url": "https://arxiv.org/pdf/2106.09685.pdf",
            },
        },
    )
    assert merged.arxiv_id == "2106.09685"
    assert "arxiv" in merged.urls


def test_hf_attach_from_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    client = HuggingFacePapersClient()

    def fake_get(url, **kwargs):
        class R:
            def raise_for_status(self) -> None:
                return None

            def json(self):
                return {
                    "githubRepo": "https://github.com/org/repo",
                    "projectPage": "https://example.com",
                    "upvotes": 3,
                    "linkedModels": [{"id": "org/model"}],
                    "linkedDatasets": [{"id": "org/ds"}],
                    "linkedSpaces": [],
                }

        return R()

    import labpilot.research_engine.intelligence.literature.clients as clients_mod

    monkeypatch.setattr(clients_mod.httpx, "get", fake_get)
    # Bypass _get_json path by patching the helper
    monkeypatch.setattr(
        clients_mod,
        "_get_json",
        lambda *a, **k: {
            "githubRepo": "https://github.com/org/repo",
            "projectPage": "https://example.com",
            "upvotes": 3,
            "linkedModels": [{"id": "org/model"}],
            "linkedDatasets": [{"id": "org/ds"}],
            "linkedSpaces": [],
        },
    )
    paper = Paper(id="arxiv:2106.09685", title="LoRA", arxiv_id="2106.09685")
    out = client.attach(paper)
    assert out.github_urls == ["https://github.com/org/repo"]
    assert "org/ds" in out.datasets
    assert out.payload["hf"]["models"] == ["org/model"]


def test_catalog_skips_redownload(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = PaperCatalogStore(ctx.knowledge_dir, ctx.competition)
    paper = Paper(
        id="arxiv:1904.08779",
        title="SpecAugment",
        abstract="masking",
        citations=10,
        relevance=0.8,
    )
    store.save_paper(paper)
    store.save_pdf(paper.id, b"%PDF-1.4 fake")
    assert store.has(paper.id)
    assert store.has_pdf(paper.id)
    # Second save without refresh keeps first version.
    v1 = store.load_pdf_path(paper.id)
    store.save_pdf(paper.id, b"%PDF-1.4 other")
    assert store.load_pdf_path(paper.id) == v1


def test_chain_uses_cache_for_existing_ids(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = PaperCatalogStore(ctx.knowledge_dir, ctx.competition)
    cached = Paper(
        id="arxiv:1",
        title="Cached Paper",
        abstract="We propose a novel CNN for birds.",
        citations=5,
        relevance=0.7,
        arxiv_id="0001.00001",
        payload={"openalex": {"id": "https://openalex.org/Wcached"}},
    )
    store.save_paper(cached)

    calls = {"n": 0}

    def fake_search(queries, limit=40):
        calls["n"] += 1
        return [
            Paper(id="arxiv:1", title="Cached Paper", relevance=0.9, citations=5),
            Paper(id="arxiv:2", title="New Paper", abstract="Introducing Mixup.", relevance=0.6),
        ]

    enrich_ids: list[str] = []

    provider = ChainedLiteratureProvider(
        catalog=store,
        download_pdfs=False,
        search_fn=fake_search,
    )

    def enrich(paper, catalog, refresh):
        enrich_ids.append(paper.id)
        return paper

    provider._enrich_chain = enrich  # type: ignore[method-assign]
    out = provider.search("birds", context=ctx, limit=10)
    assert calls["n"] == 1
    assert {p.id for p in out} == {"arxiv:1", "arxiv:2"}
    # Only the new id should hit the enrich chain.
    assert enrich_ids == ["arxiv:2"]


def test_build_literature_query_from_competition_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    run = ctx.runs_dir / "run1"
    run.mkdir()
    (run / "competition.json").write_text(
        '{"slug":"birdclef-2026","title":"BirdCLEF 2026","tags":["audio","birds"]}',
        encoding="utf-8",
    )
    queries = build_literature_query(ctx, llm_client=None)
    assert queries
    assert "BirdCLEF" in queries[0] or "birds" in queries[0].lower()


def test_paper_agent_emits_paper_knowledge() -> None:
    agent = PaperAnalyzerAgent()
    out = agent.run(
        StructuredContext(
            competition="birdclef-2026",
            text="We propose SpecAugment. Limitations include speech-domain tuning.",
            data={"techniques": ["SpecAugment"], "paper_id": "arxiv:x", "title": "SpecAugment"},
        )
    )
    assert isinstance(out, PaperKnowledge)
    assert out.techniques == ["SpecAugment"]
    assert out.paper_id == "arxiv:x"


def test_paper_agent_llm_path() -> None:
    class _Client:
        def complete(self, system: str, user: str) -> str:
            return (
                '{"paper_id":"p1","title":"T","contributions":["c"],"methods":["m"],'
                '"limitations":[],"ideas_worth_testing":["try m"],"techniques":["EMA"],'
                '"datasets_used":[],"benchmarks":[],"code_urls":[],"confidence":0.8,'
                '"grounded_in":"abstract"}'
            )

    agent = PaperAnalyzerAgent(llm_client=_Client())
    out = agent.run(StructuredContext(text="paper body"))
    assert isinstance(out, PaperKnowledge)
    assert out.techniques == ["EMA"]
    assert out.contributions == ["c"]


def test_paper_analyzer_with_fake_literature(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class FakeLit:
        notes: list[str] = []

        def search(self, query, *, context, limit=40):
            return [
                Paper(
                    id="arxiv:1904.08779",
                    title="SpecAugment",
                    abstract="We propose time and frequency masking for ASR.",
                    citations=100,
                    relevance=0.9,
                    github_urls=["https://github.com/example/specaugment"],
                )
            ]

    analyzer = PaperAnalyzer(literature=FakeLit(), llm_client=None, persist=True)
    result = analyzer.analyze(ctx)
    assert result.analyzer == "papers"
    assert len(result.items) == 1
    art = result.items[0]
    assert art.type is ResearchArtifactType.PAPER
    assert "SpecAugment" in art.title or art.techniques
    assert any("paper extraction: llm" in n for n in result.notes)


def test_paper_analyzer_empty_search(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class EmptyLit:
        notes = ["semantic scholar: no results."]

        def search(self, query, *, context, limit=40):
            return []

    result = PaperAnalyzer(literature=EmptyLit(), persist=False).analyze(ctx)
    assert result.items == []
    assert any("No papers found" in n for n in result.notes)


def test_get_json_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from labpilot.research_engine.intelligence.literature import clients as clients_mod

    calls = {"n": 0}
    sleeps: list[float] = []

    class FakeResp:
        def __init__(self, status: int, payload=None):
            self.status_code = status
            self.headers = {"Retry-After": "0.01"}
            self._payload = payload or {"ok": True}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "err",
                    request=httpx.Request("GET", "https://example.com"),
                    response=httpx.Response(self.status_code),
                )

        def json(self):
            return self._payload

    def fake_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return FakeResp(429)
        return FakeResp(200, {"data": []})

    monkeypatch.setattr(clients_mod.httpx, "get", fake_get)
    monkeypatch.setattr(clients_mod.time, "sleep", lambda s: sleeps.append(s))
    out = clients_mod._get_json("https://api.semanticscholar.org/graph/v1/paper/search")
    assert out == {"data": []}
    assert calls["n"] == 3
    assert sleeps  # backed off at least once
    # Retry-After 0.01 is floored to the configured minimum wait.
    assert all(s >= clients_mod._HTTP_MIN_WAIT_S for s in sleeps)


def test_discover_falls_back_to_openalex_on_s2_failure(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class BoomS2:
        def search(self, query, *, limit=40):
            raise httpx.HTTPStatusError(
                "429",
                request=httpx.Request("GET", "https://api.semanticscholar.org/x"),
                response=httpx.Response(429),
            )

    class EmptyArxiv:
        def search(self, query, *, limit=40):
            return []

        def attach_pdf_meta(self, paper):
            return paper

        def download_pdf(self, arxiv_id):
            return b""

    class OkOpenAlex:
        def search(self, query, *, limit=40):
            return [
                Paper(
                    id="openalex:W1",
                    title="Cell Tracking DL",
                    abstract="A method.",
                    year=2024,
                    citations=12,
                    relevance=0.9,
                )
            ]

        def enrich(self, paper):
            return paper

    class NoopAttach:
        def attach_pdf_meta(self, paper):
            return paper

        def download_pdf(self, arxiv_id):
            return b""

        def attach(self, paper):
            return paper

    provider = ChainedLiteratureProvider(
        semantic_scholar=BoomS2(),  # type: ignore[arg-type]
        openalex=OkOpenAlex(),  # type: ignore[arg-type]
        arxiv=EmptyArxiv(),  # type: ignore[arg-type]
        huggingface=NoopAttach(),  # type: ignore[arg-type]
        download_pdfs=False,
        catalog=PaperCatalogStore(ctx.knowledge_dir, ctx.competition),
    )
    out = provider.search(["cell tracking"], context=ctx, limit=10)
    assert len(out) == 1
    assert out[0].title == "Cell Tracking DL"
    assert any("OpenAlex search fallback" in n for n in provider.notes)
    assert any("HTTP 429" in n for n in provider.notes)


def test_discover_merges_s2_and_arxiv_and_dedupes(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class S2:
        def search(self, query, *, limit=40):
            return [
                Paper(
                    id="doi:10.1/same",
                    title="Same Paper",
                    abstract="From S2 with cites.",
                    doi="10.1/same",
                    arxiv_id="2106.09685",
                    citations=100,
                    relevance=0.8,
                ),
                Paper(
                    id="s2:only",
                    title="S2 Only",
                    abstract="Only on Semantic Scholar.",
                    relevance=0.6,
                ),
            ]

    class Ax:
        def search(self, query, *, limit=40):
            return [
                Paper(
                    id="arxiv:2106.09685",
                    title="Same Paper",
                    abstract="From arXiv search.",
                    arxiv_id="2106.09685",
                    doi="10.1/same",
                    relevance=0.9,
                    pdf_url="https://export.arxiv.org/pdf/2106.09685.pdf",
                ),
                Paper(
                    id="arxiv:2501.00001",
                    title="Arxiv Only",
                    abstract="Only on arXiv.",
                    arxiv_id="2501.00001",
                    relevance=0.7,
                ),
            ]

        def attach_pdf_meta(self, paper):
            return paper

    provider = ChainedLiteratureProvider(
        semantic_scholar=S2(),  # type: ignore[arg-type]
        arxiv=Ax(),  # type: ignore[arg-type]
        download_pdfs=False,
        catalog=PaperCatalogStore(ctx.knowledge_dir, ctx.competition),
    )
    provider._enrich_chain = lambda paper, catalog, refresh: paper  # type: ignore[method-assign]
    out = provider.search(["cell tracking"], context=ctx, limit=10)
    ids = {p.id for p in out}
    assert "doi:10.1/same" in ids
    assert "s2:only" in ids
    assert "arxiv:2501.00001" in ids
    assert len(out) == 3  # same paper collapsed
    same = next(p for p in out if p.id == "doi:10.1/same")
    assert same.citations == 100
    assert same.pdf_url and "2106.09685" in same.pdf_url
    assert any("discovery merge:" in n for n in provider.notes)
    assert any("arxiv=2" in n for n in provider.notes)


def test_arxiv_client_search_normalizes_result() -> None:
    from datetime import datetime
    from types import SimpleNamespace

    class FakeClient:
        def results(self, search):
            yield SimpleNamespace(
                title="Quantum Dots for Imaging",
                summary="We study quantum dots.",
                authors=[SimpleNamespace(name="Ada")],
                published=datetime(2024, 3, 1, tzinfo=UTC),
                doi="10.1/qd",
                pdf_url="https://export.arxiv.org/pdf/2403.00001.pdf",
                entry_id="http://arxiv.org/abs/2403.00001v1",
                get_short_id=lambda: "2403.00001v1",
            )

    client = ArxivClient(client=FakeClient(), delay_seconds=3.0)
    papers = client.search("quantum dots", limit=5)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2403.00001"
    assert papers[0].doi == "10.1/qd"
    assert papers[0].year == 2024
    assert papers[0].id.startswith("doi:")
    assert client.delay_seconds >= 3.0


def test_discover_keeps_partial_s2_results(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    calls = {"n": 0}

    class FlakyS2:
        def search(self, query, *, limit=40):
            calls["n"] += 1
            if calls["n"] == 1:
                return [
                    Paper(id="s2:1", title="First", abstract="a" * 50, relevance=0.9)
                ]
            raise RuntimeError("HTTP 429 (rate limited)")

    class EmptyArxiv:
        def search(self, query, *, limit=40):
            return []

        def attach_pdf_meta(self, paper):
            return paper

    provider = ChainedLiteratureProvider(
        semantic_scholar=FlakyS2(),  # type: ignore[arg-type]
        arxiv=EmptyArxiv(),  # type: ignore[arg-type]
        download_pdfs=False,
        catalog=PaperCatalogStore(ctx.knowledge_dir, ctx.competition),
    )
    provider._enrich_chain = lambda paper, catalog, refresh: paper  # type: ignore[method-assign]
    out = provider.search(["q1", "q2"], context=ctx, limit=10)
    assert len(out) == 1
    assert out[0].id == "s2:1"
    assert any("discovery merge:" in n for n in provider.notes)
    assert any("semantic_scholar=1" in n for n in provider.notes)

def test_rank_score_prefers_velocity_and_decays_age() -> None:
    from labpilot.research_engine.intelligence.literature.ranking import rank_score

    as_of = 2026
    # Old classic: huge cites, low velocity after decay.
    classic = Paper(
        id="old",
        title="Classic",
        year=2017,
        citations=10_000,
        relevance=0.9,
    )
    # Recent solid paper: fewer cites but high velocity.
    recent = Paper(
        id="new",
        title="Recent",
        year=2024,
        citations=400,
        relevance=0.9,
    )
    # Same relevance — recent should win on velocity × milder age.
    assert rank_score(recent, as_of_year=as_of) > rank_score(classic, as_of_year=as_of)


def test_select_for_extract_mixes_buckets() -> None:
    from labpilot.research_engine.intelligence.literature.ranking import select_for_extract

    as_of = 2026
    papers = [
        Paper(id="r1", title="R1", year=2024, citations=50, relevance=0.95),
        Paper(id="r2", title="R2", year=2023, citations=40, relevance=0.9),
        Paper(id="r3", title="R3", year=2025, citations=10, relevance=0.85),
        Paper(id="r4", title="R4", year=2024, citations=30, relevance=0.88),
        Paper(id="f1", title="F1", year=2017, citations=5000, relevance=0.95),
        Paper(id="f2", title="F2", year=2018, citations=3000, relevance=0.9),
        Paper(id="f3", title="F3", year=2015, citations=8000, relevance=0.8),
    ]
    picked = select_for_extract(papers, limit=5, as_of_year=as_of, recent_years=3)
    assert len(picked) == 5
    buckets = {p.id: p.bucket(as_of_year=as_of) for p in picked}
    assert any(b == "recent" for b in buckets.values())
    assert any(b == "foundational" for b in buckets.values())
    # With limit=5 and 70% recent → 4 recent + 1 foundational.
    assert sum(1 for b in buckets.values() if b == "recent") == 4
    assert sum(1 for b in buckets.values() if b == "foundational") == 1


def test_paper_analyzer_notes_bucket_mix(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)

    class FakeLit:
        notes: list[str] = []

        def search(self, query, *, context, limit=40):
            return [
                Paper(
                    id="arxiv:new",
                    title="New Method",
                    abstract="We propose a 2024 trick.",
                    year=2024,
                    citations=20,
                    relevance=0.9,
                ),
                Paper(
                    id="arxiv:old",
                    title="Classic Method",
                    abstract="We introduce a foundational idea.",
                    year=2017,
                    citations=5000,
                    relevance=0.95,
                ),
            ]

    result = PaperAnalyzer(
        literature=FakeLit(), llm_client=None, persist=False, extract_limit=2
    ).analyze(ctx)
    assert len(result.items) == 2
    assert any("recent=" in n and "foundational=" in n for n in result.notes)
    buckets = {a.metadata.get("bucket") for a in result.items}
    assert "recent" in buckets
    assert "foundational" in buckets
