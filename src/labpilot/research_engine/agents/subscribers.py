"""Default bus subscribers — decoupled reactions (no peer agent calls)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.research_engine.agents.events import (
    EVIDENCE_UPDATED,
    EXPERIMENT_COMPLETED,
    EventBus,
)
from labpilot.research_engine.memory.hooks import install_experience_memory_subscriber


def _evidence_note_path(payload: dict[str, Any]) -> Path | None:
    """Resolve a writable path for an evidence refresh note."""
    competition = str(payload.get("competition") or "unknown")
    workspace_root = payload.get("workspace_root")
    if isinstance(workspace_root, str) and workspace_root:
        note_dir = Path(workspace_root) / "artifacts"
        note_dir.mkdir(parents=True, exist_ok=True)
        return note_dir / f"evidence_refresh_{competition}.json"

    paths = [p for p in (payload.get("paths") or []) if isinstance(p, str) and p]
    for raw in paths:
        root = Path(raw)
        for candidate in (root.parent, root.parent.parent, root):
            if candidate.name in {"experiment", "artifacts", "pipeline"}:
                base = candidate.parent
                break
        else:
            base = root.parent if root.suffix else root
        if base.is_dir() or base.parent.is_dir():
            note_dir = (base if base.is_dir() else base.parent) / "artifacts"
            note_dir.mkdir(parents=True, exist_ok=True)
            return note_dir / f"evidence_refresh_{competition}.json"
    return None


def install_evidence_refresh_subscriber(bus: EventBus) -> None:
    """On ExperimentCompleted, write an evidence refresh note and emit EvidenceUpdated.

    Conductor observe can pick up the note later; Experiment never calls Reflection.
    """

    def _on_experiment_completed(event: str, payload: dict[str, Any]) -> None:
        del event
        note_path = _evidence_note_path(payload)
        if note_path is None:
            return
        if str(payload.get("status") or "").lower() == "failed" or payload.get("error"):
            # Belt and braces: the publisher no longer emits this for a failed
            # run, but this note is read straight into the Conductor's observe
            # bundle, so a wrong one steers the campaign rather than merely
            # misinforming a reader. Two guards on the path that decides what
            # the system does next is worth the duplication.
            return
        note = {
            "updated_at": datetime.now(UTC).isoformat(),
            "source_event": EXPERIMENT_COMPLETED,
            "experiment_id": payload.get("experiment_id"),
            "execution_id": payload.get("execution_id"),
            "plan_id": payload.get("plan_id"),
            "competition": payload.get("competition"),
            "metrics": payload.get("metrics") or {},
            "refs": payload.get("refs") or [],
            "observe_refresh": True,
        }
        note_path.write_text(json.dumps(note, indent=2) + "\n", encoding="utf-8")
        bus.publish(
            EVIDENCE_UPDATED,
            {
                "competition": payload.get("competition"),
                "experiment_id": payload.get("experiment_id"),
                "path": str(note_path),
                "observe_refresh": True,
            },
        )

    bus.subscribe(EXPERIMENT_COMPLETED, _on_experiment_completed)


def install_default_subscribers(bus: EventBus) -> None:
    """Install evidence refresh + experience memory write hooks."""
    install_evidence_refresh_subscriber(bus)
    install_experience_memory_subscriber(bus)
