"""Keyword rule hints over competition + text corpus (uncertain by design)."""

from __future__ import annotations

import re

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import (
    MODALITY_HINTS,
    TECHNIQUE_HINTS,
    confidence_from_hits,
)
from labpilot.research_engine.memory.models import ExperienceFacet


class RulesExtractor:
    name = "rules"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        corpus_parts = [
            ctx.competition,
            str(ctx.payload.get("problem_type") or ""),
            str(ctx.payload.get("description") or ""),
            ctx.hypothesis_text,
            ctx.action,
            str(ctx.reflection.get("observation") or ""),
            str(ctx.reflection.get("likely_cause") or ""),
            " ".join(str(t) for t in (ctx.payload.get("tags") or [])),
        ]
        corpus = re.sub(r"[_\-]+", " ", " ".join(corpus_parts).lower())
        hits: list[ExperienceFacet] = []
        for label, needles in (*MODALITY_HINTS, *TECHNIQUE_HINTS):
            matched = [n for n in needles if n in corpus]
            if not matched:
                continue
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=confidence_from_hits(len(matched)),
                    evidence=list(matched),
                    source="rules",
                )
            )
        return hits
