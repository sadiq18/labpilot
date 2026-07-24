"""Repository ranking and deep-extraction selection."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from labpilot.research_engine.intelligence.repositories.models import (
    RepoCategory,
    Repository,
)

_CATEGORY_WEIGHT = {
    RepoCategory.WINNING_SOLUTION: 1.35,
    RepoCategory.BASELINE: 1.2,
    RepoCategory.TRAINING_PIPELINE: 1.1,
    RepoCategory.AUGMENTATION: 1.05,
    RepoCategory.DOMAIN_LIBRARY: 1.0,
    RepoCategory.OTHER: 0.8,
}


def repo_rank_score(repo: Repository, *, now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    stars = max(repo.stars or 0, 0)
    star_factor = 1.0 + math.log1p(stars) / 8.0
    age_factor = 1.0
    if repo.updated_at is not None:
        updated = repo.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        age_years = max((now - updated).days / 365.25, 0.0)
        age_factor = 1.0 / (1.0 + age_years / 4.0)
    return (
        repo.relevance
        * star_factor
        * _CATEGORY_WEIGHT.get(repo.primary_category, 0.8)
        * age_factor
    )


def rank_repositories(repos: list[Repository]) -> list[Repository]:
    return sorted(repos, key=repo_rank_score, reverse=True)


def select_for_extract(repos: list[Repository], *, limit: int = 12) -> list[Repository]:
    return rank_repositories(repos)[: max(0, limit)]
