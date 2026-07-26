"""Minimal Verification Engine — pass/fail over TaskEvidence."""

from __future__ import annotations

from labpilot.research_engine.execution.schemas import TaskEvidence


def verify_evidence(evidence: TaskEvidence) -> bool:
    """Return True when evidence reports success."""
    return bool(evidence.passed) and not evidence.error
