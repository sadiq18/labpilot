"""JSON renderer — write and validate the canonical ``analyze.json`` contract.

Pydantic ``AnalysisReport`` *is* the public schema for Milestone 3 Plan 11
(no separate JSON Schema file). Validation is ``model_validate_json`` plus a
stable top-level key check used by capstone tests.
"""

from __future__ import annotations

from pathlib import Path

from labpilot.research_engine.intelligence.models import AnalysisReport

# Locked public envelope (§12.5). Nested shapes evolve via AnalysisReport fields.
PUBLIC_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "generated_at",
        "competition",
        "analyzers",
        "artifacts",
        "related_competitions",
        "papers",
        "repositories",
        "transfer_opportunities",
        "forum_knowledge",
        "knowledge_units",
        "retrieval",
        "hypothesis_recommendations",
        "techniques",
        "hypotheses",
        "suggested_experiments",
        "summary",
        "notes",
    }
)


def to_json(report: AnalysisReport) -> str:
    """Serialize the report as indented JSON (the public data contract)."""
    return report.model_dump_json(indent=2)


def validate_json(text: str) -> AnalysisReport:
    """Round-trip validate: parse text back into a typed report."""
    report = AnalysisReport.model_validate_json(text)
    assert_public_contract(report)
    return report


def assert_public_contract(report: AnalysisReport) -> None:
    """Raise ``ValueError`` if the report is missing required public keys."""
    payload = report.model_dump(mode="json")
    missing = sorted(PUBLIC_TOP_LEVEL_KEYS - set(payload))
    if missing:
        raise ValueError(f"analyze.json missing public keys: {', '.join(missing)}")
    if int(payload.get("schema_version") or 0) < 1:
        raise ValueError("analyze.json schema_version must be >= 1")
    techniques = payload.get("techniques") or {}
    for key in ("external_recommendations", "locally_validated", "unverified"):
        if key not in techniques:
            raise ValueError(f"analyze.json.techniques missing '{key}'")
    retrieval = payload.get("retrieval") or {}
    for key in ("papers", "experiments", "repositories", "discussions", "failures"):
        if key not in retrieval:
            raise ValueError(f"analyze.json.retrieval missing '{key}'")


def write_report(report: AnalysisReport, path: Path) -> Path:
    """Write ``analyze.json`` (creating parent dirs) and return the path."""
    assert_public_contract(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(report) + "\n")
    return path
