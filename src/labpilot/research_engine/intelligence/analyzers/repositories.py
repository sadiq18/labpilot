"""RepositoryAnalyzer — GitHub collect, structured extract, and local diff."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.llm.client import LLMClient
from labpilot.research_engine.intelligence.analyzers.base import BaseAnalyzer
from labpilot.research_engine.intelligence.knowledge import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.repository_analyzer import (
    RepositoryAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.repositories.differ import RepoDiffer
from labpilot.research_engine.intelligence.repositories.local_profile import (
    LocalCodeProfiler,
)
from labpilot.research_engine.intelligence.repositories.models import (
    RepoKnowledge,
    Repository,
    TransferOpportunity,
)
from labpilot.research_engine.intelligence.repositories.provider import (
    GitHubRepositoryProvider,
    RepositoryProvider,
    repositories_from_settings,
)
from labpilot.research_engine.intelligence.repositories.query import build_repo_queries
from labpilot.research_engine.intelligence.repositories.ranking import select_for_extract

logger = logging.getLogger("labpilot.research_engine.intelligence.analyzers.repositories")

DEFAULT_SEARCH_LIMIT = 30
DEFAULT_EXTRACT_LIMIT = 12


class RepositoryAnalyzer(BaseAnalyzer):
    name = "repositories"
    default_enabled = True

    def __init__(
        self,
        *,
        provider: RepositoryProvider | None = None,
        llm_client: LLMClient | None = None,
        differ: RepoDiffer | None = None,
        local_profiler: LocalCodeProfiler | None = None,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        extract_limit: int = DEFAULT_EXTRACT_LIMIT,
        competitions_dir: Path | None = None,
        persist: bool = True,
    ) -> None:
        self.provider = provider
        self.llm_client = llm_client
        self.differ = differ or RepoDiffer()
        self.local_profiler = local_profiler or LocalCodeProfiler()
        self.search_limit = search_limit
        self.extract_limit = extract_limit
        self.competitions_dir = competitions_dir
        self.persist = persist
        self._llm_explicit = llm_client is not None

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        self._maybe_attach_llm_client()
        notes: list[str] = []
        provider = self.provider or repositories_from_settings(
            knowledge_dir=context.knowledge_dir,
            competition=context.competition,
        )
        plan = build_repo_queries(
            context,
            llm_client=self.llm_client,
            competitions_dir=self.competitions_dir,
        )
        notes.append(
            "repository queries: "
            + repr([(q.category.value, q.query) for q in plan.queries])
        )
        try:
            repos = provider.search(plan, context=context, limit=self.search_limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repository search failed: %s", exc)
            repos = []
            notes.append(f"github search: unavailable — {exc}")
        if isinstance(provider, GitHubRepositoryProvider):
            notes.extend(provider.notes)
        if not repos:
            return ResearchArtifacts(
                analyzer=self.name,
                notes=[*notes, "No repositories found (search empty or soft-failed)."],
            )

        selected = select_for_extract(repos, limit=self.extract_limit)
        agent = RepositoryAnalyzerAgent(llm_client=self.llm_client)
        knowledge: list[RepoKnowledge] = []
        llm_count = 0
        for repo in selected:
            card = self._extract(agent, context, repo)
            knowledge.append(card)
            llm_count += int(agent.last_used_llm)
        notes.append(
            f"repository extraction: total={len(knowledge)}, llm={llm_count}, "
            f"rule_engine={len(knowledge) - llm_count}."
        )

        local = self.local_profiler.profile(context)
        transfers: list[TransferOpportunity] = []
        if local is None:
            notes.append("repository diff: local_unavailable (catalog-only knowledge).")
        else:
            transfers = self.differ.compare(local, knowledge)
            notes.append(
                f"repository diff: {len(transfers)} transfer opportunity(s) "
                f"from {len(local.files_scanned)} local file(s)."
            )

        by_repo: dict[str, list[TransferOpportunity]] = {}
        for transfer in transfers:
            by_repo.setdefault(transfer.repo_id, []).append(transfer)
        repo_by_id = {repo.id: repo for repo in selected}
        artifacts = [
            knowledge_to_artifact(
                context,
                repo_by_id[card.repo_id],
                card,
                by_repo.get(card.repo_id, []),
            )
            for card in knowledge
            if card.repo_id in repo_by_id
        ]
        if self.persist:
            notes.extend(self._persist(context, artifacts))
        return ResearchArtifacts(
            analyzer=self.name,
            items=artifacts,
            notes=notes,
            techniques=list(
                dict.fromkeys(technique for card in knowledge for technique in card.techniques)
            ),
            opportunities=[transfer.summary for transfer in transfers],
            transfers=[transfer.model_dump(mode="json") for transfer in transfers],
        )

    def _extract(
        self,
        agent: RepositoryAnalyzerAgent,
        context: AnalyzeContext,
        repo: Repository,
    ) -> RepoKnowledge:
        parts = [f"README:\n{repo.readme_excerpt}"]
        for path, text in repo.file_texts.items():
            parts.append(f"\nFILE {path}:\n{text}")
        result = agent.run(
            StructuredContext(
                competition=context.competition,
                text="\n".join(parts)[:120_000],
                data={
                    "repo_id": repo.id,
                    "full_name": repo.full_name,
                    "dependencies": repo.dependencies,
                    "interesting_files": repo.key_files,
                    "has_readme": bool(repo.readme_excerpt),
                },
            )
        )
        card = (
            result
            if isinstance(result, RepoKnowledge)
            else RepoKnowledge.model_validate(result.model_dump())
        )
        # Identity and file grounding come from the deterministic provider, not LLM output.
        card.repo_id = repo.id
        card.full_name = repo.full_name
        if not card.dependencies:
            card.dependencies = list(repo.dependencies)
        grounded_files = [
            path for path in card.interesting_files if path in set(repo.key_files)
        ]
        card.interesting_files = grounded_files or list(repo.key_files)
        return card

    def _maybe_attach_llm_client(self) -> None:
        if self._llm_explicit or self.llm_client is not None:
            return
        try:
            from labpilot.config import load_config
            from labpilot.llm.client import resolve_llm_client

            self.llm_client = resolve_llm_client(load_config().llm)
        except Exception:
            self.llm_client = None

    def _persist(
        self,
        context: AnalyzeContext,
        artifacts: list[ResearchArtifact],
    ) -> list[str]:
        try:
            store = KnowledgeStore(context.knowledge_dir, context.competition)
            for artifact in artifacts:
                store.upsert_artifact(artifact)
            return [f"Persisted {len(artifacts)} repository artifact(s) to knowledge.db."]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repository artifact persist failed: %s", exc)
            return [f"persist: soft-fail — {exc}"]


def knowledge_to_artifact(
    context: AnalyzeContext,
    repo: Repository,
    knowledge: RepoKnowledge,
    transfers: list[TransferOpportunity],
) -> ResearchArtifact:
    summary = transfers[0].summary if transfers else (
        "; ".join(knowledge.techniques[:3]) or repo.description or repo.full_name
    )
    meta = {
        "repository": repo.model_dump(mode="json"),
        "knowledge": knowledge.model_dump(mode="json"),
        "transfers": [transfer.model_dump(mode="json") for transfer in transfers],
        "feature_recipes": [
            recipe.model_dump(mode="json") for recipe in knowledge.feature_recipes
        ],
    }
    return ResearchArtifact(
        id=f"repo:{repo.full_name.lower()}",
        type=ResearchArtifactType.REPOSITORY,
        source="github",
        title=repo.full_name,
        summary=summary[:500],
        techniques=knowledge.techniques,
        models=knowledge.architecture,
        references=[repo.url, *repo.linked_paper_ids],
        confidence=knowledge.confidence,
        competition_slug=context.competition,
        metadata=meta,
    )


def repo_dict_for_report(artifact: ResearchArtifact) -> dict[str, Any] | None:
    if artifact.type is not ResearchArtifactType.REPOSITORY:
        return None
    meta = artifact.metadata or {}
    repo = meta.get("repository") if isinstance(meta.get("repository"), dict) else {}
    knowledge = meta.get("knowledge") if isinstance(meta.get("knowledge"), dict) else {}
    return {
        "id": artifact.id,
        "title": artifact.title,
        "url": repo.get("url"),
        "summary": artifact.summary,
        "stars": repo.get("stars"),
        "primary_category": repo.get("primary_category"),
        "techniques": artifact.techniques,
        "architecture": knowledge.get("architecture") or artifact.models,
        "loss": knowledge.get("loss") or [],
        "augmentation": knowledge.get("augmentation") or [],
        "training_tricks": knowledge.get("training_tricks") or [],
        "interesting_files": knowledge.get("interesting_files") or [],
        "confidence": artifact.confidence,
    }
