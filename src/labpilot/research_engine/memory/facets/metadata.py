"""Competition / experiment metadata facet signals."""

from __future__ import annotations

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import CONF_METADATA, CONF_TECHNIQUE_FIELD
from labpilot.research_engine.memory.models import ExperienceFacet


class MetadataExtractor:
    name = "metadata"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        hits: list[ExperienceFacet] = []
        problem_type = ctx.payload.get("problem_type")
        if problem_type:
            label = str(problem_type).lower().replace(" ", "_")
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=CONF_METADATA,
                    evidence=[str(problem_type)],
                    source="metadata",
                )
            )
        technique = ctx.payload.get("technique")
        if technique:
            label = str(technique).lower().replace(" ", "_")
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=CONF_TECHNIQUE_FIELD,
                    evidence=[str(technique)],
                    source="metadata",
                )
            )
        return hits
