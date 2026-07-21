"""Analyzer plugin interface (design §3.2).

An analyzer turns an ``AnalyzeContext`` into ``ResearchArtifacts``. It must
soft-fail (return empty artifacts + notes) rather than raise, so one broken
source never takes down a whole ``research analyze`` run.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from labpilot.research_engine.intelligence.models import AnalyzeContext, ResearchArtifacts


@runtime_checkable
class Analyzer(Protocol):
    """One pluggable research-intelligence content type."""

    name: str
    # Stable id: "competition", "papers", "repositories", "experiments",
    # "dataset", "discussions", …
    default_enabled: bool  # DiscussionAnalyzer starts False until a provider ships

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        """Read cache / M2 / call providers. Soft-fail → empty artifacts + notes."""
        ...


class BaseAnalyzer:
    """Convenience base: sets ``name`` / ``default_enabled`` and an empty result.

    Subclasses override :meth:`analyze`. The ``Independence rule`` (§3.2) holds:
    an analyzer never calls another analyzer — only its own providers, caches,
    and read-only execution libraries.
    """

    name: str = ""
    default_enabled: bool = True

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:  # pragma: no cover - abstract
        raise NotImplementedError

    def _empty(self, *notes: str) -> ResearchArtifacts:
        return ResearchArtifacts(analyzer=self.name, notes=list(notes))
