"""Plan 7 repository intelligence tests (no live GitHub calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.intelligence.analyzers.repositories import (
    RepositoryAnalyzer,
)
from labpilot.research_engine.intelligence.models import AnalyzeContext, ResearchArtifacts
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry
from labpilot.research_engine.intelligence.repositories.cache import RepoCatalogStore
from labpilot.research_engine.intelligence.repositories.clients import GitHubClient
from labpilot.research_engine.intelligence.repositories.differ import RepoDiffer
from labpilot.research_engine.intelligence.repositories.models import (
    LocalCodeProfile,
    RepoCategory,
    RepoKnowledge,
    RepoSearchPlan,
    RepoSearchQuery,
    Repository,
)
from labpilot.research_engine.intelligence.repositories.provider import (
    GitHubRepositoryProvider,
    parse_dependencies,
    select_key_paths,
)
from labpilot.research_engine.intelligence.repositories.query import build_repo_queries


def _ctx(tmp_path: Path) -> AnalyzeContext:
    return AnalyzeContext(
        competition="birdclef-2026",
        runs_dir=tmp_path / "runs",
        knowledge_dir=tmp_path / "knowledge",
    )


def _repo() -> Repository:
    return Repository(
        id="github:owner/bird",
        full_name="owner/bird",
        url="https://github.com/owner/bird",
        description="Bird classifier",
        stars=100,
        categories=[RepoCategory.BASELINE],
        primary_category=RepoCategory.BASELINE,
        readme_excerpt="Uses EfficientNet with focal loss, Mixup and EMA.",
        key_files=["train.py", "requirements.txt"],
        file_texts={
            "train.py": "loss = FocalLoss(); use_mixup=True; ema=True",
            "requirements.txt": "torch>=2\ntimm==1.0\n",
        },
        dependencies=["torch", "timm"],
        relevance=0.9,
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_github_search_normalization() -> None:
    client = GitHubClient(min_interval_s=0)
    repo = client._normalize_search(
        {
            "full_name": "Owner/Bird",
            "html_url": "https://github.com/Owner/Bird",
            "description": "Bird model",
            "stargazers_count": 42,
            "topics": ["birdclef"],
            "language": "Python",
            "updated_at": "2026-01-01T00:00:00Z",
            "default_branch": "main",
        },
        category=RepoCategory.BASELINE,
        rank_hint=0.8,
    )
    assert repo is not None
    assert repo.id == "github:owner/bird"
    assert repo.primary_category is RepoCategory.BASELINE
    assert repo.stars == 42


def test_repo_catalog_reuses_entry(tmp_path: Path) -> None:
    store = RepoCatalogStore(tmp_path / "knowledge", "birdclef")
    store.save_repo(_repo())
    assert store.load_repo("github:owner/bird") is not None
    assert [repo.id for repo in store.list_repos()] == ["github:owner/bird"]
    first = store.store.latest("repositories", "catalog__github_owner_bird")
    store.save_repo(_repo())
    assert store.store.latest("repositories", "catalog__github_owner_bird") == first


def test_key_path_selection_and_dependency_parse() -> None:
    selected = select_key_paths(
        [
            "README.md",
            "train.py",
            "src/loss.py",
            "requirements.txt",
            "docs/guide.md",
            ".github/workflows/test.yml",
        ]
    )
    assert "train.py" in selected
    assert "requirements.txt" in selected
    assert ".github/workflows/test.yml" not in selected
    assert parse_dependencies({"requirements.txt": "torch>=2\n# x\ntimm==1\n"}) == [
        "torch",
        "timm",
    ]


def test_query_planner_llm_returns_typed_plan(tmp_path: Path) -> None:
    class LLM:
        def complete(self, system: str, user: str) -> str:
            return (
                '{"queries":[{"category":"baseline",'
                '"query":"birdclef baseline language:Python"}]}'
            )

    plan = build_repo_queries(_ctx(tmp_path), llm_client=LLM())  # type: ignore[arg-type]
    assert plan.queries == [
        RepoSearchQuery(
            category=RepoCategory.BASELINE,
            query="birdclef baseline language:Python",
        )
    ]


def test_query_planner_drops_overconstrained_llm_queries(tmp_path: Path) -> None:
    class LLM:
        def complete(self, system: str, user: str) -> str:
            return (
                '{"queries":[{"category":"baseline","query":'
                '"\\"cell tracking\\" \\"development\\" competition solution '
                'language:Python"}]}'
            )

    plan = build_repo_queries(_ctx(tmp_path), llm_client=LLM())  # type: ignore[arg-type]
    # Sanitizer rejects the stacked-phrase query and falls back to seed.
    assert plan.queries
    assert all('"' not in q.query or q.query.count('"') <= 2 for q in plan.queries)
    assert any("birdclef" in q.query or "baseline" in q.query for q in plan.queries)


def test_provider_retries_broader_queries_when_primary_empty(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    calls: list[str] = []

    class Client:
        def search_repositories(self, query, *, category, limit):
            calls.append(query)
            if "language:Python" in query and "birdclef" not in query.lower():
                return []
            if "birdclef" in query.lower() or query.count(" ") <= 4:
                repo = _repo().model_copy(deep=True)
                # Avoid network in targeted fetch by returning already-cached style.
                return [repo]
            return []

        def get_repo(self, full_name):
            return {
                "stargazers_count": 100,
                "topics": [],
                "language": "Python",
                "default_branch": "main",
            }

        def get_readme(self, full_name):
            return "readme"

        def get_tree(self, full_name, ref):
            return ["train.py"]

        def get_file(self, full_name, path, *, ref):
            return "print('hi')"

    provider = GitHubRepositoryProvider(client=Client(), catalog=None)  # type: ignore[arg-type]
    # Primary plan intentionally returns nothing for these queries.
    plan = RepoSearchPlan(
        queries=[
            RepoSearchQuery(
                category=RepoCategory.BASELINE,
                query='"cell tracking" "development" competition solution language:Python',
            )
        ]
    )
    out = provider.search(plan, context=ctx, limit=5)
    assert out
    assert any("broader fallback" in note for note in provider.notes)
    assert len(calls) > 1


def test_provider_uses_cache_without_targeted_fetch(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    store = RepoCatalogStore(ctx.knowledge_dir, ctx.competition)
    store.save_repo(_repo())

    class Client:
        def search_repositories(self, query, *, category, limit):
            candidate = _repo().model_copy(deep=True)
            candidate.file_texts = {}
            return [candidate]

        def get_repo(self, full_name):
            raise AssertionError("cached repo should not refetch")

    provider = GitHubRepositoryProvider(client=Client(), catalog=store)  # type: ignore[arg-type]
    plan = RepoSearchPlan(
        queries=[RepoSearchQuery(category=RepoCategory.BASELINE, query="bird baseline")]
    )
    out = provider.search(plan, context=ctx)
    assert len(out) == 1
    assert out[0].file_texts["train.py"]
    assert any("cache_hits=1" in note for note in provider.notes)


def test_repo_differ_emits_effort_and_gain() -> None:
    local = LocalCodeProfile(loss=["cross entropy"], files_scanned=["train.py"])
    remote = RepoKnowledge(
        repo_id="github:owner/bird",
        full_name="owner/bird",
        loss=["focal loss"],
        interesting_files=["loss.py"],
    )
    transfers = RepoDiffer().compare(local, [remote])
    assert transfers
    assert transfers[0].effort.value == "20m"
    assert transfers[0].expected_gain.value == "medium"
    assert "focal loss" in transfers[0].summary


def test_repository_analyzer_and_orchestrator_merge(tmp_path: Path) -> None:
    class Provider:
        def search(self, plan, *, context, limit):
            return [_repo()]

    class Profiler:
        def profile(self, context):
            return LocalCodeProfile(
                loss=["cross entropy"],
                files_scanned=["train.py"],
            )

    analyzer = RepositoryAnalyzer(
        provider=Provider(),  # type: ignore[arg-type]
        local_profiler=Profiler(),  # type: ignore[arg-type]
        persist=False,
    )
    analyzer._llm_explicit = True
    registry = AnalyzerRegistry()
    registry.register(analyzer)
    report = AnalyzeOrchestrator(registry).analyze(_ctx(tmp_path))
    assert len(report.repositories) == 1
    assert report.repositories[0]["title"] == "owner/bird"
    assert report.transfer_opportunities
    assert report.summary["repository_count"] == 1
    assert report.summary["transfer_count"] >= 1


def test_orchestrator_accepts_transfer_only_emission(tmp_path: Path) -> None:
    class Analyzer:
        name = "transfer"
        default_enabled = True

        def analyze(self, context):
            return ResearchArtifacts(
                analyzer=self.name,
                transfers=[{"repo_id": "github:o/r", "summary": "Try EMA"}],
            )

    registry = AnalyzerRegistry()
    registry.register(Analyzer())
    report = AnalyzeOrchestrator(registry).analyze(_ctx(tmp_path))
    assert report.transfer_opportunities[0]["summary"] == "Try EMA"
