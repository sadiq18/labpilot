"""Research Intelligence Platform (Milestone 3).

A plugin pipeline over content types: analyzers emit ``ResearchArtifacts``, the
orchestrator merges them into an ``AnalysisReport`` and persists the canonical
``knowledge/<slug>/research/reports/analyze.json`` contract.

Import hygiene: this package may import ``common`` utilities and read-only
execution libraries, but must never import ``labpilot.cli``.
"""

from labpilot.research_engine.intelligence.knowledge import KnowledgeStore, RawStore
from labpilot.research_engine.intelligence.models import (
    AnalysisReport,
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.orchestrator import AnalyzeOrchestrator
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.intelligence.registry import AnalyzerRegistry

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
