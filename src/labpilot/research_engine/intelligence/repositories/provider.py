"""RepositoryProvider facade and GitHub implementation."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from labpilot.research_engine.intelligence.models import AnalyzeContext
from labpilot.research_engine.intelligence.repositories.cache import RepoCatalogStore
from labpilot.research_engine.intelligence.repositories.clients import GitHubClient
from labpilot.research_engine.intelligence.repositories.models import (
    RepoSearchPlan,
    Repository,
)
from labpilot.research_engine.intelligence.repositories.query import broaden_repo_queries
from labpilot.research_engine.intelligence.repositories.ranking import rank_repositories

logger = logging.getLogger("labpilot.research_engine.intelligence.repositories.provider")

DEFAULT_SEARCH_LIMIT = 30
_KEY_FILE_LIMIT = 12


class RepositoryProvider(ABC):
    @abstractmethod
    def search(
        self,
        plan: RepoSearchPlan,
        *,
        context: AnalyzeContext,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Repository]:
        ...


class GitHubRepositoryProvider(RepositoryProvider):
    def __init__(
        self,
        *,
        client: GitHubClient | None = None,
        catalog: RepoCatalogStore | None = None,
    ) -> None:
        self.client = client or GitHubClient()
        self.catalog = catalog
        self.notes: list[str] = []

    def search(
        self,
        plan: RepoSearchPlan,
        *,
        context: AnalyzeContext,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[Repository]:
        self.notes = []
        catalog = self.catalog or RepoCatalogStore(
            context.knowledge_dir, context.competition
        )
        if not plan.queries:
            self.notes.append("github discovery: empty search plan.")
            return []

        found: list[Repository] = []
        per_query = max(3, limit // max(len(plan.queries), 1))
        empty_queries = 0
        for item in plan.queries:
            try:
                hits = self.client.search_repositories(
                    item.query,
                    category=item.category,
                    limit=per_query,
                )
                found.extend(hits)
                if not hits:
                    empty_queries += 1
            except Exception as exc:  # noqa: BLE001 - provider soft-fail
                self.notes.append(
                    f"github search: soft-fail for {item.category.value} — {_short_err(exc)}"
                )

        # Primary plan can be over-constrained (LLM stacked quoted phrases).
        if not found:
            broad = broaden_repo_queries(context.competition)
            self.notes.append(
                f"github discovery: primary plan empty "
                f"({empty_queries}/{len(plan.queries)} zero-hit queries) — "
                "retrying broader fallback queries."
            )
            per_query = max(3, limit // max(len(broad.queries), 1))
            for item in broad.queries:
                try:
                    found.extend(
                        self.client.search_repositories(
                            item.query,
                            category=item.category,
                            limit=per_query,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.notes.append(
                        f"github search fallback: soft-fail for {item.category.value} "
                        f"— {_short_err(exc)}"
                    )

        candidates = rank_repositories(_dedupe(found))[:limit]
        if not candidates:
            cached = catalog.list_repos()
            if cached:
                self.notes.append("github discovery empty — using cached repository catalog.")
                return rank_repositories(cached)[:limit]
            self.notes.append("github discovery: no repositories found.")
            return []

        enriched: list[Repository] = []
        cache_hits = 0
        for candidate in candidates:
            cached = None if context.refresh else catalog.load_repo(candidate.id)
            if cached is not None:
                cache_hits += 1
                enriched.append(_merge_repo(cached, candidate))
                continue
            current = self._fetch_targeted(candidate, catalog, refresh=context.refresh)
            catalog.save_repo(current, refresh=context.refresh)
            enriched.append(current)

        self.notes.append(
            f"github discovery: hits={len(found)}, unique={len(candidates)}, "
            f"cache_hits={cache_hits}."
        )
        return rank_repositories(enriched)[:limit]

    def _fetch_targeted(
        self,
        repo: Repository,
        catalog: RepoCatalogStore,
        *,
        refresh: bool,
    ) -> Repository:
        updated = repo.model_copy(deep=True)
        branch = str(
            (updated.payload.get("github") or {}).get("default_branch") or "main"
        )
        try:
            meta = self.client.get_repo(updated.full_name)
            updated.stars = meta.get("stargazers_count", updated.stars)
            updated.topics = [str(v) for v in meta.get("topics") or updated.topics]
            updated.language = meta.get("language") or updated.language
            branch = str(meta.get("default_branch") or branch)
            updated.payload.setdefault("github", {})["default_branch"] = branch
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"github meta: {updated.full_name} — {_short_err(exc)}")

        try:
            readme = self.client.get_readme(updated.full_name)
            updated.readme_excerpt = readme[:40_000]
            catalog.save_text(
                updated.id, "README.md", updated.readme_excerpt, refresh=refresh
            )
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"github readme: {updated.full_name} — {_short_err(exc)}")

        paths: list[str] = []
        try:
            paths = self.client.get_tree(updated.full_name, branch)
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"github tree: {updated.full_name} — {_short_err(exc)}")

        selected = select_key_paths(paths)
        for path in selected:
            try:
                text = self.client.get_file(updated.full_name, path, ref=branch)
                updated.file_texts[path] = text
                catalog.save_text(updated.id, path, text, refresh=refresh)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(
                    f"github file: {updated.full_name}/{path} — {_short_err(exc)}"
                )
        updated.key_files = list(updated.file_texts)
        updated.dependencies = parse_dependencies(updated.file_texts)
        updated.linked_paper_ids = _linked_papers(updated.readme_excerpt)
        return updated


def repositories_from_settings(
    *,
    knowledge_dir: Path | None = None,
    competition: str | None = None,
) -> GitHubRepositoryProvider:
    from labpilot.config import Settings

    settings = Settings()
    catalog = (
        RepoCatalogStore(knowledge_dir, competition)
        if knowledge_dir is not None and competition
        else None
    )
    return GitHubRepositoryProvider(
        client=GitHubClient(token=getattr(settings, "github_token", "") or ""),
        catalog=catalog,
    )


def select_key_paths(paths: list[str]) -> list[str]:
    """Pick high-signal ML/config files from a capped Git tree."""
    scored: list[tuple[int, str]] = []
    for path in paths:
        lower = path.lower()
        name = lower.rsplit("/", 1)[-1]
        depth = lower.count("/")
        score = 0
        if name in {"requirements.txt", "pyproject.toml", "environment.yml", "setup.py"}:
            score += 20
        if any(token in name for token in ("train", "model", "loss", "augment", "config")):
            score += 14
        if lower.endswith((".py", ".yaml", ".yml", ".toml")):
            score += 4
        if lower.endswith(".md") and depth == 0 and name != "readme.md":
            score += 2
        score -= depth
        if score > 0 and not any(
            part in lower for part in ("site-packages/", "node_modules/", ".github/")
        ):
            scored.append((score, path))
    return [path for _, path in sorted(scored, key=lambda x: (-x[0], x[1]))[:_KEY_FILE_LIMIT]]


def parse_dependencies(file_texts: dict[str, str]) -> list[str]:
    deps: list[str] = []
    for path, text in file_texts.items():
        lower = path.lower()
        if not any(
            token in lower
            for token in ("requirements", "pyproject.toml", "environment.yml", "setup.py")
        ):
            continue
        for line in text.splitlines():
            stripped = line.strip().strip('"').strip("'")
            if not stripped or stripped.startswith(("#", "[", "-r", "python")):
                continue
            match = re.match(r"-?\s*([A-Za-z0-9_.-]+)", stripped)
            if match:
                name = match.group(1).lower()
                if name not in {"dependencies", "dev-dependencies", "name", "version"}:
                    deps.append(name)
    return list(dict.fromkeys(deps))[:100]


def _linked_papers(text: str) -> list[str]:
    ids = re.findall(r"arxiv\.org/(?:abs|pdf)/([A-Za-z0-9./-]+)", text, flags=re.I)
    return [f"arxiv:{value.removesuffix('.pdf')}" for value in dict.fromkeys(ids)]


def _dedupe(repos: list[Repository]) -> list[Repository]:
    by_name: dict[str, Repository] = {}
    for repo in repos:
        key = repo.full_name.lower()
        if key in by_name:
            by_name[key] = _merge_repo(by_name[key], repo)
        else:
            by_name[key] = repo
    return list(by_name.values())


def _merge_repo(existing: Repository, incoming: Repository) -> Repository:
    updated = existing.model_copy(deep=True)
    updated.relevance = max(existing.relevance, incoming.relevance)
    updated.stars = max(existing.stars or 0, incoming.stars or 0)
    updated.categories = list(
        dict.fromkeys([*existing.categories, *incoming.categories])
    )
    if incoming.relevance > existing.relevance:
        updated.primary_category = incoming.primary_category
    if not updated.description:
        updated.description = incoming.description
    return updated


def _short_err(exc: Exception) -> str:
    if hasattr(exc, "response") and getattr(exc, "response") is not None:
        status = getattr(exc.response, "status_code", None)
        if status:
            return f"HTTP {status}"
    return str(exc).strip()[:120] or type(exc).__name__
