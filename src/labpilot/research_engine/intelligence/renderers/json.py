"""JSON renderer — write and validate the canonical ``analyze.json`` contract."""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.intelligence.models import AnalysisReport


def to_json(report: AnalysisReport) -> str:
    """Serialize the report as indented JSON (the public data contract)."""
    return report.model_dump_json(indent=2)


def validate_json(text: str) -> AnalysisReport:
    """Round-trip validate: parse text back into a typed report."""
    return AnalysisReport.model_validate_json(text)


def write_report(report: AnalysisReport, path: Path) -> Path:
    """Write ``analyze.json`` (creating parent dirs) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(report) + "\n")
    return path
