"""Typed envelope and adapters around reflection pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.artifacts.base import ARTIFACT_SCHEMA_IDS, ArtifactMeta, ArtifactRef
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.reflection.pipeline import run_reflection

SCHEMA_ID = ARTIFACT_SCHEMA_IDS["reflection"]


class ReflectionResult(BaseModel):
    """Normalized view of one reflection run.

    Database rows remain the source of record; this model is a stable API shape
    over the pipeline's return dict (and optional JSON projection).
    """

    schema_id: str = SCHEMA_ID
    competition: str
    execution_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    assessment: dict[str, Any] = Field(default_factory=dict)
    belief: dict[str, Any] = Field(default_factory=dict)
    hypothesis: dict[str, Any] = Field(default_factory=dict)
    lesson: dict[str, Any] = Field(default_factory=dict)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    synthesis: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


def _envelope_from_raw(competition: str, raw: dict[str, Any]) -> ReflectionResult:
    """Build a :class:`ReflectionResult` from a ``run_reflection`` return dict."""
    assessment = raw.get("assessment")
    if assessment is None:
        assessment_dict: dict[str, Any] = {}
    elif hasattr(assessment, "model_dump"):
        assessment_dict = assessment.model_dump(mode="json")
    elif isinstance(assessment, dict):
        assessment_dict = assessment
    else:
        assessment_dict = {"value": str(assessment)}

    evidence = raw.get("evidence") or {}
    execution_id = None
    if isinstance(evidence, dict):
        execution_id = evidence.get("execution_id") or evidence.get("experiment_id")

    return ReflectionResult(
        competition=competition,
        execution_id=execution_id if isinstance(execution_id, str) else None,
        evidence=evidence if isinstance(evidence, dict) else {},
        assessment=assessment_dict,
        belief=raw.get("belief") if isinstance(raw.get("belief"), dict) else {},
        hypothesis=raw.get("hypothesis") if isinstance(raw.get("hypothesis"), dict) else {},
        lesson=raw.get("lesson") if isinstance(raw.get("lesson"), dict) else {},
        claims=list(raw.get("claims") or []) if isinstance(raw.get("claims"), list) else [],
        synthesis=raw.get("synthesis") if isinstance(raw.get("synthesis"), dict) else {},
        recommendation=(
            raw.get("recommendation")
            if isinstance(raw.get("recommendation"), dict)
            else {}
        ),
        raw=raw,
    )


def run_and_wrap(
    knowledge_dir: Path,
    competition: str,
    *,
    execution_id: str | None = None,
    workspace_path: Path | str | None = None,
    plan_id: str | None = None,
    hypothesis_id: str | None = None,
    llm_client: Any | None = None,
    persist: bool = True,
    write_projection: bool = True,
    produced_by: str = "reflect",
) -> tuple[ReflectionResult, ArtifactRef]:
    """Run reflection, wrap the result, and optionally write a JSON projection.

    Returns the typed envelope and an :class:`ArtifactRef` (``path`` set when a
    projection file was written).
    """
    raw = run_reflection(
        knowledge_dir,
        competition,
        execution_id=execution_id,
        workspace_path=workspace_path,
        plan_id=plan_id,
        hypothesis_id=hypothesis_id,
        llm_client=llm_client,
        persist=persist,
    )
    result = _envelope_from_raw(competition, raw)
    paths = ResearchPaths(knowledge_dir, competition).ensure()
    out_dir = paths.root / "reflection"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.execution_id or "latest"
    path = out_dir / f"reflection_{stem}.json"
    if write_projection and persist:
        slim = result.model_copy(update={"raw": {}})
        path.write_text(slim.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = ArtifactMeta(schema_id=SCHEMA_ID, produced_by=produced_by)
    ref = ArtifactRef(
        kind="reflection",
        id=f"reflection:{stem}",
        schema_id=SCHEMA_ID,
        path=str(path) if path.is_file() else None,
        competition=competition,
    )
    return result, ref


def read_projection(
    knowledge_dir: Path,
    competition: str,
    *,
    execution_id: str | None = None,
) -> ReflectionResult | None:
    """Load a reflection JSON projection, or ``None`` if missing."""
    paths = ResearchPaths(knowledge_dir, competition).ensure()
    stem = execution_id or "latest"
    path = paths.root / "reflection" / f"reflection_{stem}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ReflectionResult.model_validate(data)
