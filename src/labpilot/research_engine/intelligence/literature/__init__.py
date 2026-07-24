"""Literature provider chain (Plan 6 — Paper Research Engine)."""

from labpilot.research_engine.intelligence.literature.cache import PaperCatalogStore
from labpilot.research_engine.intelligence.literature.models import Paper, PaperKnowledge
from labpilot.research_engine.intelligence.literature.provider import (
    ChainedLiteratureProvider,
    LiteratureProvider,
    literature_from_settings,
)
from labpilot.research_engine.intelligence.literature.query import build_literature_query
from labpilot.research_engine.intelligence.literature.ranking import (
    select_for_extract,
)

__all__ = [
    "ChainedLiteratureProvider",
    "LiteratureProvider",
    "Paper",
    "PaperCatalogStore",
    "PaperKnowledge",
    "build_literature_query",
    "literature_from_settings",
    "select_for_extract",
]
