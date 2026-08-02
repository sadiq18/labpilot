"""FacetPipeline — run source extractors, merge, log confidence histogram."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from labpilot.research_engine.memory.facets.base import FacetExtractor
from labpilot.research_engine.memory.facets.code import CodeExtractor
from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.dataset import DatasetExtractor
from labpilot.research_engine.memory.facets.merge import merge_facet_hits
from labpilot.research_engine.memory.facets.metadata import MetadataExtractor
from labpilot.research_engine.memory.facets.paper import PaperExtractor
from labpilot.research_engine.memory.facets.result import ResultExtractor
from labpilot.research_engine.memory.facets.rules import RulesExtractor
from labpilot.research_engine.memory.models import ExperienceFacet

logger = logging.getLogger(__name__)

# Bands for Stage-4 trigger analysis (LLM only when mid/low share hurts seed/inspect).
_LOW = 0.55
_MID = 0.75


def default_extractors() -> list[FacetExtractor]:
    return [
        MetadataExtractor(),
        RulesExtractor(),
        CodeExtractor(),
        DatasetExtractor(),
        PaperExtractor(),
        ResultExtractor(),
    ]


class FacetPipeline:
    """Artifact-aware facet extraction with soft-fail per source."""

    def __init__(self, extractors: Sequence[FacetExtractor] | None = None) -> None:
        self._extractors = list(extractors) if extractors is not None else default_extractors()

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        hits: list[ExperienceFacet] = []
        for extractor in self._extractors:
            try:
                hits.extend(extractor.extract(ctx))
            except Exception:  # noqa: BLE001 — never fail experience write path
                logger.exception("facet extractor %s failed", getattr(extractor, "name", "?"))
        merged = merge_facet_hits(hits)
        log_confidence_histogram(merged, competition=ctx.competition)
        return merged


def log_confidence_histogram(
    facets: list[ExperienceFacet],
    *,
    competition: str,
) -> dict[str, Any]:
    """Log band counts so Stage 4 can wait for real mid/low-share evidence."""
    low = mid = high = 0
    for facet in facets:
        if facet.confidence < _LOW:
            low += 1
        elif facet.confidence < _MID:
            mid += 1
        else:
            high += 1
    summary = {
        "competition": competition,
        "facet_count": len(facets),
        "low": low,
        "mid": mid,
        "high": high,
        "max_confidence": max((f.confidence for f in facets), default=0.0),
        "min_confidence": min((f.confidence for f in facets), default=0.0),
    }
    logger.info(
        "experience_facet_confidence_histogram competition=%s count=%s "
        "low=%s mid=%s high=%s min=%.2f max=%.2f",
        competition,
        summary["facet_count"],
        low,
        mid,
        high,
        summary["min_confidence"],
        summary["max_confidence"],
    )
    return summary
