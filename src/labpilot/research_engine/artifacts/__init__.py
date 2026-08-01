"""Typed adapters for durable research artifacts.

Callers (CLI or orchestrators) read and write through this package. Underlying
stores and models live in sibling packages; those packages must not import here.

Import direction::

    callers → artifacts → {intelligence, planner, execution, evidence, reflection}
"""

from __future__ import annotations

from labpilot.research_engine.artifacts.analysis import read_analysis, write_analysis
from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.artifacts.evidence import EvidenceArtifacts
from labpilot.research_engine.artifacts.execution import ExecutionArtifacts
from labpilot.research_engine.artifacts.plan import PlanArtifacts
from labpilot.research_engine.artifacts.reflection import ReflectionResult, run_and_wrap
from labpilot.research_engine.artifacts.submission import SubmissionRecord, write_submission_record

__all__ = [
    "ARTIFACT_SCHEMA_IDS",
    "ArtifactMeta",
    "ArtifactRef",
    "EvidenceArtifacts",
    "ExecutionArtifacts",
    "PlanArtifacts",
    "ReflectionResult",
    "SubmissionRecord",
    "read_analysis",
    "run_and_wrap",
    "write_analysis",
    "write_submission_record",
]
