"""Optional paper-title / abstract facet signals (no live literature fetch)."""

from __future__ import annotations

import re

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import (
    CONF_PAPER,
    MODALITY_HINTS,
    TECHNIQUE_HINTS,
    confidence_from_hits,
)
from labpilot.research_engine.memory.models import ExperienceFacet


class PaperExtractor:
    name = "paper"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        texts = list(ctx.paper_texts)
        papers = ctx.payload.get("papers")
        if isinstance(papers, list):
            for item in papers:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(str(item.get("title") or ""))
                    texts.append(str(item.get("abstract") or item.get("summary") or ""))
        corpus = re.sub(r"[_\-]+", " ", " ".join(texts).lower()).strip()
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
                    confidence=min(CONF_PAPER, confidence_from_hits(len(matched))),
                    evidence=list(matched),
                    source="paper",
                )
            )
        return hits
