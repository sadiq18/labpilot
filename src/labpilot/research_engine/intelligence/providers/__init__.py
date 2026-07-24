"""Capability providers for Research Intelligence analyzers (design §3.5).

Providers isolate optional / external sources (related competitions, winning
solutions, …) behind an explicit ``CapabilityResult`` so analyzers never pretend
to know a field they did not resolve — and never special-case HTML scrapes.
"""

from labpilot.research_engine.intelligence.providers.capability import (
    CapabilityResult,
    CompetitionProfile,
    ExternalDataPolicy,
    InferenceLimits,
    RelatedCompetition,
)
from labpilot.research_engine.intelligence.providers.related import (
    RelatedCompetitionLookup,
    RelatedCompetitionProvider,
    SeriesRelatedCompetitionProvider,
)
from labpilot.research_engine.intelligence.providers.winning_solutions import (
    NullWinningSolutionProvider,
    WinningSolutionProvider,
)

__all__ = [
    "CapabilityResult",
    "CompetitionProfile",
    "ExternalDataPolicy",
    "InferenceLimits",
    "NullWinningSolutionProvider",
    "RelatedCompetition",
    "RelatedCompetitionLookup",
    "RelatedCompetitionProvider",
    "SeriesRelatedCompetitionProvider",
    "WinningSolutionProvider",
]
