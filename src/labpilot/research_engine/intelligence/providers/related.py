"""Related / similar competition recall (design §3.5).

Deterministic signals first (series prefix, shared tags, modality/metric
overlap). An optional metadata fetcher verifies previous-edition slugs and
searches for peers via the official Kaggle API — never HTML scrape.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from labpilot.competition.models import CompetitionMetadata, CompetitionSpec
from labpilot.research_engine.intelligence.models import AnalyzeContext
from labpilot.research_engine.intelligence.providers.capability import (
    CapabilityResult,
    RelatedCompetition,
)

logger = logging.getLogger("labpilot.research_engine.intelligence.providers.related")

_YEAR_SUFFIX = re.compile(r"^(?P<base>.+?)[-_](?P<year>\d{4})$", re.IGNORECASE)


@runtime_checkable
class RelatedCompetitionProvider(Protocol):
    def find(
        self,
        competition: str,
        *,
        context: AnalyzeContext,
        spec: CompetitionSpec | None = None,
    ) -> RelatedCompetitionLookup:
        """Return related competitions with an explicit capability status."""
        ...


class CompetitionMetadataFetcher(Protocol):
    """Minimal read surface used by the related-comp provider (matches KaggleClient)."""

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None: ...


class RelatedCompetitionLookup(BaseModel):
    """Capability envelope plus the typed related-comp list."""

    capability: CapabilityResult
    related: list[RelatedCompetition] = Field(default_factory=list)


def series_base(slug: str) -> str | None:
    """``birdclef-2026`` → ``birdclef``; returns None when no year suffix."""
    match = _YEAR_SUFFIX.match(slug.strip())
    return match.group("base").lower() if match else None


def previous_edition_slugs(slug: str, *, years_back: int = 5) -> list[str]:
    """Candidate previous-edition slugs derived from a trailing year."""
    match = _YEAR_SUFFIX.match(slug.strip())
    if not match:
        return []
    base = match.group("base")
    year = int(match.group("year"))
    # Prefer the separator that appears between base and year in the input.
    rest = slug[len(base) : len(base) + 1]
    sep = rest if rest in {"-", "_"} else "-"
    return [f"{base}{sep}{year - offset}" for offset in range(1, years_back + 1)]


class SeriesRelatedCompetitionProvider:
    """Official-API related-comp recall with series-prefix heuristics.

    When no ``metadata_fetcher`` is configured, returns ``unavailable`` rather
    than inventing competitions. With a fetcher, verifies year-decremented
    previous editions and scores optional search hits by tag overlap.
    """

    def __init__(
        self,
        metadata_fetcher: CompetitionMetadataFetcher | None = None,
        *,
        max_similar: int = 5,
        search: CompetitionSearch | None = None,
    ) -> None:
        self.metadata_fetcher = metadata_fetcher
        self.max_similar = max_similar
        self.search = search

    def find(
        self,
        competition: str,
        *,
        context: AnalyzeContext,
        spec: CompetitionSpec | None = None,
    ) -> RelatedCompetitionLookup:
        del context  # reserved for future cache / seed YAML paths
        if self.metadata_fetcher is None and self.search is None:
            return RelatedCompetitionLookup(
                capability=CapabilityResult(
                    available=False,
                    status="unavailable",
                    reason="No competition metadata fetcher configured for related-comp recall.",
                )
            )

        related: list[RelatedCompetition] = []
        seen: set[str] = {competition.lower()}

        if self.metadata_fetcher is not None:
            for candidate in previous_edition_slugs(competition):
                if candidate.lower() in seen:
                    continue
                meta = self._safe_fetch(candidate)
                if meta is None:
                    continue
                seen.add(candidate.lower())
                related.append(
                    RelatedCompetition(
                        slug=meta.slug or candidate,
                        title=meta.title or candidate,
                        relation="previous_edition",
                        score=0.95,
                        rationale=f"Same series as '{competition}' (prior year).",
                        tags_overlap=_tag_overlap(spec.tags if spec else [], meta.tags),
                    )
                )

        query = series_base(competition) or (spec.tags[0] if spec and spec.tags else competition)
        if self.search is not None:
            similar_count = 0
            for meta in self._safe_search(query):
                slug = (meta.slug or "").lower()
                if not slug or slug in seen:
                    continue
                seen.add(slug)
                overlap = _tag_overlap(spec.tags if spec else [], meta.tags)
                relation, score, rationale = _classify_similar(
                    competition, query, meta, overlap, spec
                )
                if relation == "previous_edition":
                    # Already covered by year walk when possible; keep if missed.
                    related.append(
                        RelatedCompetition(
                            slug=meta.slug,
                            title=meta.title or meta.slug,
                            relation=relation,
                            score=score,
                            rationale=rationale,
                            tags_overlap=overlap,
                        )
                    )
                    continue
                related.append(
                    RelatedCompetition(
                        slug=meta.slug,
                        title=meta.title or meta.slug,
                        relation=relation,
                        score=score,
                        rationale=rationale,
                        tags_overlap=overlap,
                    )
                )
                similar_count += 1
                if similar_count >= self.max_similar:
                    break

        if not related:
            return RelatedCompetitionLookup(
                capability=CapabilityResult(
                    available=False,
                    status="unavailable",
                    reason=f"No related competitions found for '{competition}'.",
                )
            )
        return RelatedCompetitionLookup(
            capability=CapabilityResult(
                available=True,
                status="ok",
                reason=f"Found {len(related)} related competition(s).",
            ),
            related=related,
        )

    def _safe_fetch(self, slug: str) -> CompetitionMetadata | None:
        assert self.metadata_fetcher is not None
        try:
            return self.metadata_fetcher.fetch_competition_metadata(slug)
        except Exception:
            logger.warning("Related-comp metadata fetch failed for '%s'.", slug, exc_info=True)
            return None

    def _safe_search(self, query: str) -> list[CompetitionMetadata]:
        assert self.search is not None
        try:
            return list(self.search.search_competitions(query))
        except Exception:
            logger.warning("Related-comp search failed for %r.", query, exc_info=True)
            return []


class CompetitionSearch(Protocol):
    def search_competitions(self, query: str) -> list[CompetitionMetadata]: ...


def _tag_overlap(left: list[str], right: list[str]) -> list[str]:
    left_norm = {t.strip().lower() for t in left if t.strip()}
    overlap: list[str] = []
    seen: set[str] = set()
    for tag in right:
        key = tag.strip().lower()
        if key and key in left_norm and key not in seen:
            seen.add(key)
            overlap.append(tag)
    return overlap


def _classify_similar(
    competition: str,
    query: str,
    meta: CompetitionMetadata,
    overlap: list[str],
    spec: CompetitionSpec | None,
) -> tuple[str, float, str]:
    slug = (meta.slug or "").lower()
    base = series_base(competition)
    other_base = series_base(slug)
    if base and other_base and base == other_base and slug != competition.lower():
        return (
            "previous_edition",
            0.9,
            f"Same series as '{competition}'.",
        )

    metric_raw = (meta.evaluation_metric_raw or "").strip().lower()
    spec_metric = ""
    if spec and spec.evaluation_metric is not None:
        spec_metric = (spec.evaluation_metric.name or "").strip().lower()
    if metric_raw and spec_metric and metric_raw == spec_metric:
        return (
            "similar_metric",
            0.7,
            f"Shares evaluation metric '{spec.evaluation_metric.name}'.",
        )

    if overlap:
        return (
            "similar_domain",
            min(0.85, 0.5 + 0.1 * len(overlap)),
            f"Overlapping tags: {', '.join(overlap)}.",
        )

    if query and query.lower() in slug:
        return (
            "similar_domain",
            0.55,
            f"Slug shares series/token '{query}'.",
        )

    return ("other", 0.4, f"Returned by search for '{query}'.")
