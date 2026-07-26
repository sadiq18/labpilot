import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from labpilot.cli import main as cli_main
from labpilot.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.intelligence.context import build_context, normalize_competition
from labpilot.research_engine.intelligence.knowledge import KnowledgeHub, KnowledgeStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import (
    AnalyzerRegistry,
    UnknownAnalyzerError,
    build_default_registry,
)
from labpilot.research_engine.intelligence.renderers.json import (
    to_json,
    validate_json,
    write_report,
)

runner = CliRunner()


def _plain(text: str) -> str:
    """Strip ANSI codes and collapse whitespace.

    Typer/rich renders --help inside a panel and folds long option tokens across
    lines at narrow terminal widths (which differ between local and CI). Collapsing
    whitespace makes flag assertions robust to that wrapping.
    """
    without_ansi = re.sub(r"\x1b\[[0-9;]*m", "", text)
    return re.sub(r"\s+", "", without_ansi)


class FakeAnalyzer:
    def __init__(self, name: str, *, default_enabled: bool = True, items=None, notes=None):
        self.name = name
        self.default_enabled = default_enabled
        self._items = items or []
        self._notes = notes or []

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        return ResearchArtifacts(analyzer=self.name, items=self._items, notes=self._notes)


class BoomAnalyzer:
    name = "boom"
    default_enabled = True

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        raise RuntimeError("provider exploded")


def _artifact(id_: str = "paper:1") -> ResearchArtifact:
    return ResearchArtifact(
        id=id_,
        type=ResearchArtifactType.PAPER,
        source="semantic_scholar",
        title="Attention Is All You Need",
        techniques=["attention"],
        confidence=0.9,
    )


# --- ResearchArtifact schema ------------------------------------------------


def test_research_artifact_round_trip():
    original = _artifact()
    restored = ResearchArtifact.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.type is ResearchArtifactType.PAPER


def test_research_artifact_defaults_are_present():
    art = ResearchArtifact(id="x", type=ResearchArtifactType.NOTE, source="user")
    assert art.confidence == 0.5
    assert art.techniques == [] and art.references == []
    # migration aliases exist but default empty
    assert art.concepts == [] and art.evidence == [] and art.payload == {}


def test_research_artifact_confidence_bounds():
    with pytest.raises(ValueError):
        ResearchArtifact(id="x", type=ResearchArtifactType.NOTE, source="user", confidence=1.5)


# --- Registry ---------------------------------------------------------------


def test_registry_register_and_list_preserves_order():
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("competition"))
    reg.register(FakeAnalyzer("papers"))
    assert reg.names() == ["competition", "papers"]


def test_registry_rejects_duplicate_and_empty_names():
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers"))
    with pytest.raises(ValueError):
        reg.register(FakeAnalyzer("papers"))
    with pytest.raises(ValueError):
        reg.register(FakeAnalyzer(""))


def test_registry_select_default_respects_default_enabled():
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers"))
    reg.register(FakeAnalyzer("discussions", default_enabled=False))
    selected = [a.name for a in reg.select()]
    assert selected == ["papers"]


def test_registry_select_only_include_exclude():
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("competition"))
    reg.register(FakeAnalyzer("papers"))
    reg.register(FakeAnalyzer("dataset"))

    assert [a.name for a in reg.select(only="dataset")] == ["dataset"]
    assert [a.name for a in reg.select(include={"papers", "dataset"})] == ["papers", "dataset"]
    assert [a.name for a in reg.select(exclude={"dataset"})] == ["competition", "papers"]


def test_registry_unknown_names_raise():
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers"))
    with pytest.raises(UnknownAnalyzerError):
        reg.get("nope")
    with pytest.raises(UnknownAnalyzerError):
        reg.select(include={"nope"})
    with pytest.raises(ValueError):
        reg.select(only="papers", include={"papers"})


def test_default_registry_has_local_analyzers():
    # Plans 4–7 built-in analyzers.
    assert build_default_registry().names() == [
        "competition",
        "experiments",
        "dataset",
        "papers",
        "repositories",
    ]


# --- Context normalization --------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_slug",
    [
        ("birdclef-2026", "birdclef-2026"),
        ("birdclef-2026/", "birdclef-2026"),
        ("https://www.kaggle.com/competitions/birdclef-2026", "birdclef-2026"),
        ("https://www.kaggle.com/competitions/birdclef-2026/overview", "birdclef-2026"),
        ("https://www.kaggle.com/c/titanic", "titanic"),
        ("www.kaggle.com/competitions/spaceship-titanic", "spaceship-titanic"),
    ],
)
def test_normalize_competition(value, expected_slug):
    slug, url = normalize_competition(value)
    assert slug == expected_slug
    if "kaggle.com" in value:
        assert url is not None
    else:
        assert url is None


def test_normalize_competition_rejects_empty():
    with pytest.raises(ValueError):
        normalize_competition("   ")


def test_build_context_paths(tmp_path: Path):
    ctx = build_context(
        "birdclef-2026", runs_dir=tmp_path / "runs", knowledge_dir=tmp_path / "knowledge"
    )
    assert ctx.competition == "birdclef-2026"
    assert ctx.report_path == tmp_path / "knowledge/birdclef-2026/research/reports/analyze.json"


# --- Orchestrator -----------------------------------------------------------


def _ctx(tmp_path: Path) -> AnalyzeContext:
    return build_context(
        "birdclef-2026", runs_dir=tmp_path / "runs", knowledge_dir=tmp_path / "knowledge"
    )


def test_orchestrator_empty_registry_writes_stub(tmp_path: Path):
    orch = AnalyzeOrchestrator(AnalyzerRegistry())
    report = orch.analyze(_ctx(tmp_path))
    assert report.competition["slug"] == "birdclef-2026"
    assert report.analyzers == []
    assert any("No analyzers" in note for note in report.notes)


def test_orchestrator_merges_artifacts(tmp_path: Path):
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")], notes=["cache hit"]))
    reg.register(FakeAnalyzer("dataset", items=[_artifact("dataset:1")]))
    report = AnalyzeOrchestrator(reg).analyze(_ctx(tmp_path))
    assert report.analyzers == ["papers", "dataset"]
    assert {a.id for a in report.artifacts} == {"paper:1", "dataset:1"}
    assert "[papers] cache hit" in report.notes
    assert report.summary["artifact_count"] == 2
    # PAPER artifacts land in report.papers (terminal Papers count).
    assert len(report.papers) == 2
    assert report.papers[0]["id"] == "paper:1"
    assert report.summary["paper_count"] == 2


def test_micro_agent_retries_transient_llm_errors(monkeypatch) -> None:
    from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
    from labpilot.research_engine.intelligence.literature.models import PaperKnowledge

    sleeps: list[float] = []
    monkeypatch.setattr("labpilot.accessor.common.micro_agents.time.sleep", lambda s: sleeps.append(s))

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("503 UNAVAILABLE high demand")
            return (
                '{"paper_id":"p","title":"T","contributions":["c"],"methods":[],'
                '"limitations":[],"ideas_worth_testing":[],"techniques":["EMA"],'
                '"datasets_used":[],"benchmarks":[],"code_urls":[],"confidence":0.7,'
                '"grounded_in":"abstract"}'
            )

    class Agent(BaseMicroAgent):
        name = "RetryAgent"
        output_model = PaperKnowledge
        llm_max_attempts = 3
        llm_retry_delay_seconds = 0.01

        def system_prompt(self) -> str:
            return "sys"

        def user_prompt(self, context: StructuredContext) -> str:
            return "user"

        def _run_rule_engine(self, context: StructuredContext) -> PaperKnowledge:
            return PaperKnowledge(paper_id="fallback")

    client = Flaky()
    agent = Agent(llm_client=client)
    out = agent.run(StructuredContext(text="paper"))
    assert isinstance(out, PaperKnowledge)
    assert out.techniques == ["EMA"]
    assert client.calls == 3
    assert agent.last_used_llm is True
    assert len(sleeps) == 2


def test_orchestrator_soft_fails_on_analyzer_exception(tmp_path: Path):
    reg = AnalyzerRegistry()
    reg.register(BoomAnalyzer())
    reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
    report = AnalyzeOrchestrator(reg).analyze(_ctx(tmp_path))
    # boom still counted as run, but contributed a failure note, no crash
    assert "boom" in report.analyzers
    assert any("analyzer failed" in note for note in report.notes)
    assert {a.id for a in report.artifacts} == {"paper:1"}


def test_orchestrator_url_recorded(tmp_path: Path):
    ctx = build_context(
        "https://www.kaggle.com/competitions/birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )
    report = AnalyzeOrchestrator(AnalyzerRegistry()).analyze(ctx)
    assert report.competition["url"].endswith("birdclef-2026")


# --- JSON renderer ----------------------------------------------------------


def test_json_round_trip_and_write(tmp_path: Path):
    report = AnalysisReport(competition={"slug": "birdclef-2026"}, analyzers=["papers"])
    text = to_json(report)
    assert validate_json(text) == report

    path = write_report(report, tmp_path / "a/b/analyze.json")
    assert path.is_file()
    loaded = json.loads(path.read_text())
    assert loaded["competition"]["slug"] == "birdclef-2026"
    assert loaded["schema_version"] == 2


# --- CLI --------------------------------------------------------------------


def test_analyze_help_documents_flags():
    result = runner.invoke(
        cli_main.app, ["analyze", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"}
    )
    assert result.exit_code == 0
    plain = _plain(result.stdout)
    for flag in (
        "--include",
        "--exclude",
        "--format",
        "--refresh",
        "--skip-ingest",
        "--skip-hypothesize",
        "--skip-brief",
        "--fetch-kaggle",
    ):
        assert flag in plain


def test_analyze_cli_writes_stub_report(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_main, "build_default_registry", lambda: AnalyzerRegistry())
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "birdclef-2026",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    report_path = tmp_path / "knowledge/birdclef-2026/research/reports/analyze.json"
    assert report_path.is_file()
    validate_json(report_path.read_text())


def test_analyze_cli_can_skip_knowledge_ingestion(tmp_path: Path, monkeypatch):
    def _registry():
        reg = AnalyzerRegistry()
        reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
        return reg

    monkeypatch.setattr(cli_main, "build_default_registry", _registry)
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "birdclef-2026",
            "--skip-ingest",
            "--format",
            "json",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["knowledge_units"] == []


def test_ingest_cli_processes_pending_stored_artifacts(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        store.upsert_artifact(_artifact("paper:1"))

    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda _config: None)
    args = [
        "ingest",
        "birdclef-2026",
        "--knowledge-dir",
        str(knowledge_dir),
    ]
    first = runner.invoke(cli_main.app, args)
    assert first.exit_code == 0, first.stdout
    assert "1 unit(s), 1 belief(s)" in first.stdout

    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        artifacts = store.list_artifacts()
        assert KnowledgeHub(store).pending_artifacts(artifacts) == []

    second = runner.invoke(cli_main.app, args)
    assert second.exit_code == 0, second.stdout
    assert "0 pending" in second.stdout


def test_ingest_cli_generates_hypotheses_unless_skipped(tmp_path: Path, monkeypatch):
    knowledge_dir = tmp_path / "knowledge"
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        store.upsert_artifact(_artifact("paper:1"))

    monkeypatch.setattr(cli_main, "resolve_llm_client", lambda _config: None)
    base = ["ingest", "birdclef-2026", "--knowledge-dir", str(knowledge_dir)]

    skipped = runner.invoke(cli_main.app, [*base, "--skip-hypothesize"])
    assert skipped.exit_code == 0, skipped.stdout
    assert "Hypothesis generation skipped" in skipped.stdout
    assert HypothesisStore(knowledge_dir, "birdclef-2026").list() == []

    generated = runner.invoke(cli_main.app, base)
    assert generated.exit_code == 0, generated.stdout
    assert "new hypothesis generated" in generated.stdout
    assert HypothesisStore(knowledge_dir, "birdclef-2026").list()


def test_analyze_skip_ingest_also_skips_hypotheses(tmp_path: Path, monkeypatch):
    def _registry():
        reg = AnalyzerRegistry()
        reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
        return reg

    monkeypatch.setattr(cli_main, "build_default_registry", _registry)
    knowledge_dir = tmp_path / "knowledge"
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "birdclef-2026",
            "--skip-ingest",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert HypothesisStore(knowledge_dir, "birdclef-2026").list() == []
    brief = knowledge_dir / "birdclef-2026" / "research" / "reports" / "research_brief.md"
    assert not brief.is_file()


def test_analyze_persists_dataset_and_experiment_artifacts(tmp_path: Path) -> None:
    dataset = ResearchArtifact(
        id="dataset:birdclef-2026",
        type=ResearchArtifactType.DATASET,
        source="m2",
        title="dataset card",
        competition_slug="birdclef-2026",
        metadata={"modality": "audio", "row_count": 100},
    )
    experiment = ResearchArtifact(
        id="exp:12",
        type=ResearchArtifactType.EXPERIMENT,
        source="m2",
        title="exp-12",
        techniques=["Focal Loss"],
        competition_slug="birdclef-2026",
    )
    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("dataset", items=[dataset]))
    reg.register(FakeAnalyzer("experiments", items=[experiment]))
    knowledge_dir = tmp_path / "knowledge"
    ctx = build_context(
        "birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=knowledge_dir,
    )
    report = AnalyzeOrchestrator(reg, llm_client=None).analyze(ctx)
    assert report.research_brief
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        assert store.get_artifact("dataset:birdclef-2026") is not None
        assert store.get_artifact("exp:12") is not None


def test_analyze_skip_brief_writes_no_markdown(tmp_path: Path, monkeypatch) -> None:
    def _registry():
        reg = AnalyzerRegistry()
        reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
        return reg

    monkeypatch.setattr(cli_main, "build_default_registry", _registry)
    knowledge_dir = tmp_path / "knowledge"
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "birdclef-2026",
            "--skip-brief",
            "--knowledge-dir",
            str(knowledge_dir),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    brief = knowledge_dir / "birdclef-2026" / "research" / "reports" / "research_brief.md"
    assert not brief.is_file()
    assert HypothesisStore(knowledge_dir, "birdclef-2026").list()


def test_analyze_fetch_kaggle_runs_three_fetches(tmp_path: Path) -> None:
    from labpilot.research_engine.intelligence.fetch.models import FetchResult
    from labpilot.research_engine.intelligence.models import ResearchArtifactType

    calls: list[tuple[frozenset[str], dict]] = []

    class FakeFetchService:
        def fetch(self, competition, *, sources, knowledge_dir, refresh=False, **kwargs):
            calls.append((frozenset(sources), dict(kwargs)))
            artifact = ResearchArtifact(
                id=f"fetched:{len(calls)}",
                type=(
                    ResearchArtifactType.DISCUSSION
                    if "discussions" in sources
                    else ResearchArtifactType.REPOSITORY
                ),
                source="kaggle",
                title=f"fetched-{len(calls)}",
                competition_slug=competition,
                techniques=["Mixup"] if "kernels" in sources else [],
            )
            with KnowledgeStore(knowledge_dir, competition) as store:
                store.upsert_artifact(artifact)
            return FetchResult(
                competition=competition,
                sources=sorted(sources),
                written=1,
                skipped_existing=0,
                fetched=1,
                artifact_ids=[artifact.id],
                notes=[],
            )

    reg = AnalyzerRegistry()
    reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
    knowledge_dir = tmp_path / "knowledge"
    ctx = build_context(
        "birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=knowledge_dir,
    )
    report = AnalyzeOrchestrator(
        reg,
        llm_client=None,
        fetch_kaggle=True,
        kaggle_fetch_service=FakeFetchService(),  # type: ignore[arg-type]
    ).analyze(ctx)

    assert len(calls) == 3
    assert calls[0][0] == frozenset({"kernels"})
    assert calls[0][1].get("kernel_sort") == "voteCount"
    assert calls[0][1].get("limit") == 5
    assert calls[1][0] == frozenset({"kernels"})
    assert calls[1][1].get("kernel_sort") == "scoreDescending"
    assert calls[2][0] == frozenset({"discussions"})
    assert any("[fetch-kaggle]" in note for note in report.notes)
    with KnowledgeStore(knowledge_dir, "birdclef-2026") as store:
        ids = {a.id for a in store.list_artifacts()}
        assert "paper:1" in ids
        assert "fetched:1" in ids
        assert "fetched:3" in ids


def test_analyze_cli_single_analyzer_and_json_format(tmp_path: Path, monkeypatch):
    def _registry():
        reg = AnalyzerRegistry()
        reg.register(FakeAnalyzer("papers", items=[_artifact("paper:1")]))
        reg.register(FakeAnalyzer("dataset", items=[_artifact("dataset:1")]))
        return reg

    monkeypatch.setattr(cli_main, "build_default_registry", _registry)
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "papers",
            "birdclef-2026",
            "--format",
            "json",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["analyzers"] == ["papers"]
    assert [a["id"] for a in payload["artifacts"]] == ["paper:1"]


def test_analyze_cli_unknown_analyzer_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_main, "build_default_registry", lambda: AnalyzerRegistry())
    result = runner.invoke(
        cli_main.app,
        [
            "analyze",
            "nope",
            "birdclef-2026",
            "--knowledge-dir",
            str(tmp_path / "knowledge"),
        ],
    )
    assert result.exit_code == 1
    assert "Unknown analyzer" in result.stdout


def test_analyze_cli_rejects_bad_format(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_main, "build_default_registry", lambda: AnalyzerRegistry())
    result = runner.invoke(cli_main.app, ["analyze", "birdclef-2026", "--format", "html"])
    assert result.exit_code != 0
