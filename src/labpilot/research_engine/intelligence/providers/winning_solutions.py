"""Winning-solution capability providers (design §3.5).

v1 rule (locked): prefer official API when it exposes solutions; otherwise
report ``status: unavailable`` via :class:`NullWinningSolutionProvider`.
**Do not HTML-scrape** writeups in Milestone 3 — a future provider may swap in
without rewriting ``CompetitionAnalyzer``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.intelligence.models import AnalyzeContext
from labpilot.research_engine.intelligence.providers.capability import CapabilityResult


@runtime_checkable
class WinningSolutionProvider(Protocol):
    def fetch(self, competition: str, *, context: AnalyzeContext) -> CapabilityResult:
        """Return winning-solution artifacts, or an explicit unavailable/error."""
        ...


class NullWinningSolutionProvider:
    """Default v1 provider — honest unavailable, never fabricates writeups."""

    def fetch(self, competition: str, *, context: AnalyzeContext) -> CapabilityResult:
        return CapabilityResult(
            available=False,
            status="unavailable",
            reason="Not available through configured provider.",
        )
