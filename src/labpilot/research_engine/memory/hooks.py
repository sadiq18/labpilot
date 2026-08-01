"""Write-only experience persistence hooks (no Conductor / task scheduling)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.memory.extractor import ExperienceExtractor
from labpilot.research_engine.memory.models import ExperienceRecord
from labpilot.workspace import MARKER_NAME, load_workspace

logger = logging.getLogger(__name__)


def _resolve_knowledge_dir(payload: dict[str, Any]) -> Path | None:
    raw = payload.get("knowledge_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser().resolve()
    workspace_root = payload.get("workspace_root")
    if isinstance(workspace_root, str) and workspace_root.strip():
        root = Path(workspace_root).expanduser().resolve()
        marker = root / MARKER_NAME
        if marker.is_file():
            try:
                return load_workspace(marker).knowledge_dir
            except (OSError, ValueError):
                pass
        candidate = root / "knowledge"
        if candidate.is_dir():
            return candidate.resolve()
    return None


def persist_experience_from_completion(
    payload: dict[str, Any],
) -> ExperienceRecord | None:
    """Extract + upsert an Experience Record from a completion payload.

    Failures are logged and swallowed so experiment pipelines never fail because
    of memory writes. Does not enqueue tasks or touch Conductor policy.
    """
    try:
        competition = str(payload.get("competition") or "").strip()
        if not competition:
            logger.debug("experience hook skipped: missing competition")
            return None
        knowledge_dir = _resolve_knowledge_dir(payload)
        if knowledge_dir is None:
            logger.debug("experience hook skipped: missing knowledge_dir")
            return None

        experiment_id = payload.get("experiment_id")
        execution_id = payload.get("execution_id")
        if not experiment_id and not execution_id:
            logger.debug("experience hook skipped: missing experiment/execution id")
            return None

        experiment_payload: dict[str, Any] = {
            "experiment_id": experiment_id,
            "execution_id": execution_id,
            "plan_id": payload.get("plan_id"),
            "competition": competition,
            "status": payload.get("status") or "completed",
            "metrics": payload.get("metrics") or {},
            "git_commit": payload.get("git_commit"),
            "files_changed": payload.get("files_changed") or [],
            "description": payload.get("description") or payload.get("action") or "",
            "hypothesis_id": payload.get("hypothesis_id"),
        }
        comparison = payload.get("comparison")
        if isinstance(comparison, dict):
            experiment_payload["comparison"] = comparison

        reflection = payload.get("reflection")
        workspace_root = payload.get("workspace_root")

        extractor = ExperienceExtractor(knowledge_dir)
        try:
            return extractor.extract(
                competition=competition,
                experiment=experiment_payload,
                experiment_id=str(experiment_id) if experiment_id else None,
                execution_id=str(execution_id) if execution_id else None,
                plan_id=str(payload["plan_id"]) if payload.get("plan_id") else None,
                hypothesis_id=(
                    str(payload["hypothesis_id"]) if payload.get("hypothesis_id") else None
                ),
                reflection=reflection if isinstance(reflection, dict) else None,
                comparison=comparison if isinstance(comparison, dict) else None,
                workspace_path=Path(workspace_root) if workspace_root else None,
                persist=True,
            )
        finally:
            extractor.close()
    except Exception:  # noqa: BLE001 — never fail the experiment pipeline
        logger.exception("experience memory write hook failed")
        return None


def install_experience_memory_subscriber(bus: Any) -> None:
    """Subscribe to ExperimentCompleted — write-only ExperienceStore upsert."""
    from labpilot.research_engine.agents.events import EXPERIMENT_COMPLETED

    def _on_experiment_completed(event: str, payload: dict[str, Any]) -> None:
        del event
        persist_experience_from_completion(payload)

    bus.subscribe(EXPERIMENT_COMPLETED, _on_experiment_completed)
