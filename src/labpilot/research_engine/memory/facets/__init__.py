"""Artifact-aware experience facet extraction (Stage 2)."""

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.merge import merge_facet_hits
from labpilot.research_engine.memory.facets.pipeline import (
    FacetPipeline,
    default_extractors,
    log_confidence_histogram,
)

__all__ = [
    "FacetContext",
    "FacetPipeline",
    "default_extractors",
    "log_confidence_histogram",
    "merge_facet_hits",
]
