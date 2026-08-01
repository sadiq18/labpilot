"""Operator-driven experience seed manifests (auditable warm-start priors)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.research_engine.memory.models import ExperienceRecord
from labpilot.workspace import competition_data_root


def seed_dir(knowledge_dir: Path, competition: str) -> Path:
    return competition_data_root(Path(knowledge_dir), competition) / "memory" / "seeds"


def seed_manifest_path(knowledge_dir: Path, competition: str, source_slug: str) -> Path:
    safe = source_slug.strip().replace("/", "_")
    return seed_dir(knowledge_dir, competition) / f"{safe}.json"


def write_seed_manifest(
    knowledge_dir: Path,
    *,
    target_competition: str,
    source_competition: str,
    records: list[ExperienceRecord],
) -> Path:
    """Materialize an auditable seed file under the target competition workspace."""
    path = seed_manifest_path(knowledge_dir, target_competition, source_competition)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "target_competition": target_competition,
        "source_competition": source_competition,
        "seeded_at": datetime.now(UTC).isoformat(),
        "experience_ids": [r.id for r in records],
        "experiences": [r.model_dump(mode="json") for r in records],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_seeded_experience_ids(knowledge_dir: Path, competition: str) -> set[str]:
    """Union of experience ids from all seed manifests for ``competition``."""
    root = seed_dir(knowledge_dir, competition)
    if not root.is_dir():
        return set()
    ids: set[str] = set()
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for exp_id in data.get("experience_ids") or []:
            text = str(exp_id).strip()
            if text:
                ids.add(text)
    return ids
