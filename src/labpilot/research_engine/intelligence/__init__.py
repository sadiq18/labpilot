"""Research Intelligence Platform (Milestone 3).

A plugin pipeline over content types: analyzers emit ``ResearchArtifacts``, the
orchestrator merges them into an ``AnalysisReport`` and persists the canonical
``knowledge/<slug>/research/reports/analyze.json`` contract.

Import hygiene: this package may import ``common`` utilities and read-only
execution libraries, but must never import ``labpilot.cli``.

Heavy symbols are lazy-imported so ``intelligence.competition.models`` /
``intelligence.paths`` can load without pulling the orchestrator (avoids
cycles with ``experiments.hypothesis`` → ``graph`` → baseline selector).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AnalysisReport",
    "AnalyzeContext",
    "AnalyzeOrchestrator",
    "AnalyzerRegistry",
    "KnowledgeStore",
    "RawStore",
    "ResearchArtifact",
    "ResearchArtifacts",
    "ResearchArtifactType",
    "ResearchPaths",
]


def __getattr__(name: str) -> Any:
    if name in {"KnowledgeStore", "RawStore"}:
        from labpilot.research_engine.intelligence import knowledge as _knowledge

        return getattr(_knowledge, name)
    if name in {
        "AnalysisReport",
        "AnalyzeContext",
        "ResearchArtifact",
        "ResearchArtifacts",
        "ResearchArtifactType",
    }:
        from labpilot.research_engine.intelligence import models as _models

        return getattr(_models, name)
    if name == "AnalyzeOrchestrator":
        from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator

        return AnalyzeOrchestrator
    if name == "ResearchPaths":
        from labpilot.research_engine.intelligence.paths import ResearchPaths

        return ResearchPaths
    if name == "AnalyzerRegistry":
        from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

        return AnalyzerRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
