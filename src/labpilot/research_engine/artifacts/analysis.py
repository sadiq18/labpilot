"""Read/write adapters for competition analysis reports (``analyze.json``)."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.intelligence.models import AnalysisReport
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.intelligence.renderers.json import (
    validate_json,
    write_report,
)

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["competition_analysis"]

# Alias used in design docs; same model as AnalysisReport.
CompetitionAnalysis = AnalysisReport


def analysis_report_path(knowledge_dir: Path, competition: str) -> Path:
    """Return the canonical ``analyze.json`` path for a competition."""
    return ResearchPaths(knowledge_dir, competition).ensure().report_path


def write_analysis(
    report: AnalysisReport,
    knowledge_dir: Path,
    competition: str,
    *,
    path: Path | None = None,
    produced_by: str = "analyze",
) -> ArtifactRef:
    """Persist an analysis report and return an :class:`ArtifactRef`.

    Writes to ``path`` when given; otherwise the competition's default
    ``analyze.json`` location.
    """
    target = path or analysis_report_path(knowledge_dir, competition)
    write_report(report, target)
    _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
    return ArtifactRef(
        kind="competition_analysis",
        id=f"analysis:{competition}",
        schema_id=SCHEMA_ID,
        path=str(target),
        competition=competition,
    )


def read_analysis(
    knowledge_dir: Path,
    competition: str,
    *,
    path: Path | None = None,
) -> AnalysisReport | None:
    """Load an analysis report, or ``None`` if the file is missing."""
    target = path or analysis_report_path(knowledge_dir, competition)
    if not target.is_file():
        return None
    return validate_json(target.read_text(encoding="utf-8"))
