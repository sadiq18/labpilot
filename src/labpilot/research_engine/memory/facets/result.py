"""Reflection / comparison / metric-key facet signals."""

from __future__ import annotations

import re

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import (
    CONF_RESULT,
    MODALITY_HINTS,
    TECHNIQUE_HINTS,
    confidence_from_hits,
)
from labpilot.research_engine.memory.models import ExperienceFacet


class ResultExtractor:
    name = "result"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        parts = [
            str(ctx.reflection.get("observation") or ""),
            str(ctx.reflection.get("likely_cause") or ""),
            str(ctx.comparison.get("verdict") or ""),
            str(ctx.comparison.get("notes") or ""),
            " ".join(str(k) for k in (ctx.payload.get("metrics") or {})),
        ]
        corpus = re.sub(r"[_\-]+", " ", " ".join(parts).lower()).strip()
        if not corpus:
            return []
        hits: list[ExperienceFacet] = []
        for label, needles in (*MODALITY_HINTS, *TECHNIQUE_HINTS):
            matched = [n for n in needles if n in corpus]
            if not matched:
                continue
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=min(CONF_RESULT, confidence_from_hits(len(matched))),
                    evidence=list(matched),
                    source="result",
                )
            )
        return hits
