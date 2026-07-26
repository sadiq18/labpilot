"""Plan 5 — CompetitionAnalyzer + capability providers.

No live Kaggle / HTML scrape in CI: metadata and related-comp recall are
injected via fakes. NullWinningSolutionProvider must report unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

from labpilot.research_engine.intelligence.competition.models import CompetitionMetadata, CompetitionSpec, MetricSpec
from labpilot.research_engine.intelligence.analyzers.competition import CompetitionAnalyzer
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.providers.capability import CapabilityResult
from labpilot.research_engine.intelligence.providers.related import (
    SeriesRelatedCompetitionProvider,
    previous_edition_slugs,
    series_base,
)
from labpilot.research_engine.intelligence.providers.winning_solutions import (
    NullWinningSolutionProvider,
)
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry


class _FakeFetcher:
    def __init__(self, by_slug: dict[str, CompetitionMetadata]) -> None:
        self.by_slug = {k.lower(): v for k, v in by_slug.items()}
        self.calls: list[str] = []

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None:
        self.calls.append(competition)
        return self.by_slug.get(competition.lower())


class _FakeSearch:
    def __init__(self, results: list[CompetitionMetadata]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search_competitions(self, query: str) -> list[CompetitionMetadata]:
        # Simulate an API that returns a candidate set for the query; filtering
        # / scoring is the provider's job.
        self.queries.append(query)
        return list(self.results)


def _context(tmp_path: Path, competition: str = "birdclef-2026") -> AnalyzeContext:
    return AnalyzeContext(
        competition=competition,
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )


def _seed_competition_json(
    runs_dir: Path,
    *,
    competition: str = "birdclef-2026",
    run_id: str = "run-1",
    raw_html: str = "",
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    spec = CompetitionSpec(
        slug=competition,
        title="BirdCLEF 2026",
        description="Bird soundscape classification",
        evaluation_metric=MetricSpec(name="macro f1", direction="maximize", key="macro_f1"),
        problem_type="unknown",
        submission_mode="csv",
        deadline="2026-06-01T00:00:00",
        max_daily_submissions=5,
        tags=["audio", "birds"],
        raw_html=raw_html,
        data_url=f"https://www.kaggle.com/competitions/{competition}/data",
        rules_url=f"https://www.kaggle.com/competitions/{competition}/rules",
    )
    (run_dir / "competition.json").write_text(spec.model_dump_json(indent=2))
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "competition": competition,
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
                "status": "completed",
                "stages": [],
                "metadata": {},
            }
        )
    )
    return run_dir


# --- helpers ----------------------------------------------------------------


def test_series_base_and_previous_edition_slugs():
    assert series_base("birdclef-2026") == "birdclef"
    assert previous_edition_slugs("birdclef-2026", years_back=2) == [
        "birdclef-2025",
        "birdclef-2024",
    ]
    assert series_base("titanic") is None
    assert previous_edition_slugs("titanic") == []


def test_null_winning_solution_provider_is_unavailable():
    ctx = AnalyzeContext(competition="x", runs_dir=Path("."), knowledge_dir=Path("."))
    result = NullWinningSolutionProvider().fetch("x", context=ctx)
    assert result.available is False
    assert result.status == "unavailable"
    assert "Not available" in result.reason


# --- related provider -------------------------------------------------------


def test_related_provider_unavailable_without_fetcher():
    ctx = _context(Path("/tmp"))
    lookup = SeriesRelatedCompetitionProvider().find("birdclef-2026", context=ctx)
    assert lookup.capability.status == "unavailable"
    assert lookup.related == []


def test_related_provider_finds_previous_editions_and_similar():
    fetcher = _FakeFetcher(
        {
            "birdclef-2025": CompetitionMetadata(
                slug="birdclef-2025", title="BirdCLEF 2025", tags=["audio", "birds"]
            ),
            "birdclef-2024": CompetitionMetadata(
                slug="birdclef-2024", title="BirdCLEF 2024", tags=["audio"]
            ),
        }
    )
    search = _FakeSearch(
        [
            CompetitionMetadata(
                slug="rfcx-species-audio-detection",
                title="Rainforest Connection",
                tags=["audio", "birds"],
                evaluation_metric_raw="macro f1",
            ),
            CompetitionMetadata(slug="birdclef-2026", title="self"),
        ]
    )
    provider = SeriesRelatedCompetitionProvider(fetcher, search=search)
    ctx = _context(Path("/tmp"))
    spec = CompetitionSpec(
        slug="birdclef-2026",
        tags=["audio", "birds"],
        evaluation_metric=MetricSpec(name="macro f1", direction="maximize"),
    )
    lookup = provider.find("birdclef-2026", context=ctx, spec=spec)
    assert lookup.capability.status == "ok"
    relations = {r.slug: r.relation for r in lookup.related}
    assert relations["birdclef-2025"] == "previous_edition"
    assert relations["birdclef-2024"] == "previous_edition"
    assert "rfcx-species-audio-detection" in relations


# --- CompetitionAnalyzer ----------------------------------------------------


def test_competition_analyzer_builds_profile_from_cache(tmp_path: Path, monkeypatch):
    # Avoid accidental network from CompetitionParser rules scrape on miss.
    monkeypatch.setattr("labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt", lambda *a, **k: "")
    ctx = _context(tmp_path)
    _seed_competition_json(
        ctx.runs_dir,
        raw_html="External data is allowed. Pretrained weights ok.",
    )
    fetcher = _FakeFetcher(
        {
            "birdclef-2025": CompetitionMetadata(
                slug="birdclef-2025", title="BirdCLEF 2025", tags=["audio"]
            )
        }
    )

    def fake_pages(slug: str, **kwargs):
        from labpilot.research_engine.intelligence.competition.page_fetch import CompetitionPages

        text = (
            "## Overview\nBird soundscape classification.\n\n"
            "## Rules\nExternal data is allowed. Pretrained weights ok.\n"
            "Internet access is disabled for submission kernels.\n"
            "Kernels limited to 9 hours runtime on GPU.\n"
            "## Evaluation\nscore = mean Macro F1 across classes\n"
            "## Submission\nSubmit a CSV file with columns id,label.\n"
        )
        return CompetitionPages(
            slug=slug,
            overview_url="https://example.test/overview",
            rules_url="https://example.test/rules",
            overview_text="Bird soundscape classification.",
            rules_text=text,
            is_empty_shell=False,
        )

    analyzer = CompetitionAnalyzer(
        metadata_fetcher=fetcher,
        related_provider=SeriesRelatedCompetitionProvider(fetcher),
        persist=True,
        page_fetcher=fake_pages,
        llm_client=None,
    )
    result = analyzer.analyze(ctx)

    assert result.analyzer == "competition"
    profile = next(a for a in result.items if a.metadata.get("kind") == "profile")
    assert profile.id == "competition:birdclef-2026"
    assert profile.type is ResearchArtifactType.COMPETITION
    dumped = profile.metadata["profile"]
    assert dumped["title"] == "BirdCLEF 2026"
    assert dumped["metric"]["key"] == "macro_f1"
    assert dumped["submission"]["mode"] == "csv"
    assert dumped["external_data"]["status"] == "ok"
    assert dumped["external_data"]["allowed"] is True
    assert dumped["inference_limits"]["status"] == "ok"
    assert dumped["inference_limits"]["internet_allowed"] is False
    assert dumped["evaluation"]["formula"]
    assert dumped["page_enrichment_source"] == "rule_engine"
    assert dumped["winning_solutions"]["status"] == "unavailable"
    assert any(a.metadata.get("relation") == "previous_edition" for a in result.items)
    assert any("winning solutions: unavailable" in n for n in result.notes)

    db = ctx.knowledge_dir / ctx.competition / "research" / "knowledge.db"
    assert db.is_file()


def test_competition_analyzer_soft_profile_without_cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt", lambda *a, **k: "")
    ctx = _context(tmp_path, competition="spaceship-titanic")
    ctx.runs_dir.mkdir(parents=True)

    def empty_pages(slug: str, **kwargs):
        from labpilot.research_engine.intelligence.competition.page_fetch import CompetitionPages

        return CompetitionPages(
            slug=slug,
            overview_url="u",
            rules_url="r",
            overview_text="",
            rules_text="",
            is_empty_shell=True,
            source="none",
        )

    analyzer = CompetitionAnalyzer(
        persist=False, page_fetcher=empty_pages, llm_client=None
    )
    result = analyzer.analyze(ctx)
    profile = next(a for a in result.items if a.metadata.get("kind") == "profile")
    assert profile.id == "competition:spaceship-titanic"
    assert profile.metadata["profile"]["winning_solutions"]["status"] == "unavailable"
    assert any("related competitions: unavailable" in n for n in result.notes)
    assert any("page enrichment: unavailable" in n for n in result.notes)


def test_orchestrator_merges_competition_sections(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt", lambda *a, **k: "")
    ctx = _context(tmp_path)
    _seed_competition_json(ctx.runs_dir)
    fetcher = _FakeFetcher(
        {
            "birdclef-2025": CompetitionMetadata(
                slug="birdclef-2025", title="BirdCLEF 2025", tags=["audio"]
            )
        }
    )

    def empty_pages(slug: str, **kwargs):
        from labpilot.research_engine.intelligence.competition.page_fetch import CompetitionPages

        return CompetitionPages(
            slug=slug,
            overview_url="u",
            rules_url="r",
            overview_text="x" * 250,
            rules_text="External data is not permitted.",
            is_empty_shell=False,
        )

    reg = AnalyzerRegistry()
    reg.register(
        CompetitionAnalyzer(
            metadata_fetcher=fetcher,
            related_provider=SeriesRelatedCompetitionProvider(fetcher),
            persist=False,
            page_fetcher=empty_pages,
            llm_client=None,
        )
    )
    report = AnalyzeOrchestrator(reg).analyze(ctx, only="competition")
    assert report.competition["slug"] == "birdclef-2026"
    assert report.competition["title"] == "BirdCLEF 2026"
    assert report.competition["winning_solutions"]["status"] == "unavailable"
    assert any(r["slug"] == "birdclef-2025" for r in report.related_competitions)


def test_no_html_scrape_branch_in_winning_provider():
    # Guardrail: Null provider is the only default; analyzer never scrapes writeups.
    src = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    competition_py = (src / "intelligence" / "analyzers" / "competition.py").read_text()
    winning_py = (src / "intelligence" / "providers" / "winning_solutions.py").read_text()
    assert "BeautifulSoup" not in competition_py
    assert "httpx" not in competition_py
    assert "BeautifulSoup" not in winning_py
    assert "httpx" not in winning_py
    assert "NullWinningSolutionProvider" in competition_py
    assert "NullWinningSolutionProvider" in winning_py


def test_competition_analyzer_skips_kaggle_without_credentials(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("labpilot.research_engine.intelligence.competition.parser.fetch_rules_excerpt", lambda *a, **k: "")
    monkeypatch.setattr("labpilot.diagnostics.kaggle_credentials_present", lambda: False)
    ctx = _context(tmp_path, competition="biohub-cell-tracking-during-development")
    ctx.runs_dir.mkdir(parents=True)

    def empty_pages(slug: str, **kwargs):
        from labpilot.research_engine.intelligence.competition.page_fetch import CompetitionPages

        return CompetitionPages(
            slug=slug,
            overview_url="u",
            rules_url="r",
            overview_text="",
            rules_text="",
            is_empty_shell=True,
            source="none",
        )

    analyzer = CompetitionAnalyzer(
        persist=False, page_fetcher=empty_pages, llm_client=None
    )
    result = analyzer.analyze(ctx)
    assert analyzer.metadata_fetcher is None
    assert any("Kaggle credentials not found" in n for n in result.notes)
    profile = next(a for a in result.items if a.metadata.get("kind") == "profile")
    assert profile.id.startswith("competition:biohub")


def test_page_agent_rule_engine_and_llm():
    from labpilot.common.micro_agents import StructuredContext
    from labpilot.research_engine.intelligence.micro_agents.artifacts import (
        CompetitionPageExtract,
    )
    from labpilot.research_engine.intelligence.micro_agents.competition_page_analyzer import (
        CompetitionPageAnalyzerAgent,
    )

    text = (
        "## Overview\nTrack cells in 3D.\n\n"
        "## Rules\nExternal data is allowed. Internet access is disabled.\n"
        "## Evaluation\nscore = mean F1\n"
        "## Submission\nCSV file required.\n"
    )
    agent = CompetitionPageAnalyzerAgent()
    out = agent.run(StructuredContext(text=text))
    assert isinstance(out, CompetitionPageExtract)
    assert out.external_data_allowed is True
    assert out.internet_allowed is False
    assert out.submission_format == "csv"
    assert "F1" in out.evaluation_formula or "f1" in out.evaluation_formula.lower()

    class _Static:
        def complete(self, system: str, user: str) -> str:
            return (
                '{"external_data_allowed": false, "pretrained_weights_allowed": false, '
                '"external_data_notes": "no external", "runtime_notes": "", '
                '"hardware_notes": "", "internet_allowed": null, "inference_notes": "", '
                '"evaluation_formula": "HOTA", "evaluation_description": "tracking", '
                '"submission_format": "csv", "submission_columns_notes": "", '
                '"sample_submission_notes": "", "overview_summary": "cells", '
                '"other_notes": ""}'
            )

    llm_out = CompetitionPageAnalyzerAgent(llm_client=_Static()).run(
        StructuredContext(text=text)
    )
    assert llm_out.external_data_allowed is False
    assert llm_out.evaluation_formula == "HOTA"


def test_capability_result_model():
    result = CapabilityResult(available=False, status="unavailable", reason="x")
    assert result.items == []
