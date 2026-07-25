"""Result models for ``KaggleFetchService``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FetchResult(BaseModel):
    """Counts and soft-fail notes from one fetch run."""

    competition: str
    sources: list[str] = Field(default_factory=list)
    fetched: int = 0
    skipped_existing: int = 0
    written: int = 0
    pages_scanned: int = 0
    llm_enriched: int = 0
    rule_engine_enriched: int = 0
    notes: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
