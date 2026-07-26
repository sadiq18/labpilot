"""Bounded TaskContext assembled by the Research Engineer (ephemeral)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.execution.schemas import ResearchExecution, TaskEvidence
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan, ResearchTask


@dataclass
class TaskContext:
    """Platform-owned memory for one task attempt — never session chat memory."""

    plan: ResearchPlan
    task: ResearchTask
    execution: ResearchExecution
    paths: ResearchPaths
    workspace_root: Path
    competition: str
    relevant_files: list[str] = field(default_factory=list)
    prior_evidence: TaskEvidence | None = None
    runtime_target: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
