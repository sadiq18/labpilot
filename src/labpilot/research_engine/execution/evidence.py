"""Evidence file helpers under ``…/executions/E-xxx/evidence/``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.intelligence.paths import ResearchPaths


def execution_root(paths: ResearchPaths, execution_id: str) -> Path:
    return paths.executions_dir / execution_id


def evidence_dir(paths: ResearchPaths, execution_id: str) -> Path:
    return execution_root(paths, execution_id) / "evidence"


def evidence_path(paths: ResearchPaths, execution_id: str, task_id: str) -> Path:
    safe = task_id.replace("/", "_")
    return evidence_dir(paths, execution_id) / f"{safe}.json"


def ensure_execution_layout(paths: ResearchPaths, execution_id: str) -> Path:
    """Create ``executions/E-xxx/{evidence,artifacts,logs}/`` and return root."""
    root = execution_root(paths, execution_id)
    for sub in ("evidence", "artifacts", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def write_evidence(
    paths: ResearchPaths,
    evidence: TaskEvidence,
) -> Path:
    """Persist task evidence JSON; returns the written path."""
    if evidence.created_at is None:
        evidence = evidence.model_copy(update={"created_at": datetime.now(UTC)})
    path = evidence_path(paths, evidence.execution_id, evidence.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_evidence(
    paths: ResearchPaths,
    execution_id: str,
    task_id: str,
) -> TaskEvidence | None:
    path = evidence_path(paths, execution_id, task_id)
    if not path.is_file():
        return None
    return TaskEvidence.model_validate_json(path.read_text(encoding="utf-8"))
