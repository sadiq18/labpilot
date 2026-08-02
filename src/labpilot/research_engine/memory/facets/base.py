"""FacetExtractor protocol — one source, soft-fail friendly."""

from __future__ import annotations

from typing import Protocol

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.models import ExperienceFacet


class FacetExtractor(Protocol):
    """Emit evidence-backed facet hits from a single artifact source."""

    name: str

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        """Return zero or more hits; never raise into the pipeline."""
        ...
