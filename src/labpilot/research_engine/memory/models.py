"""Experience Record models — transferable research memory SoR."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ExperienceOutcome = Literal["success", "fail"]
FacetSource = Literal["metadata", "rules", "legacy"]


class ExperienceFacet(BaseModel):
    """Evidence-backed facet hit — hints are not treated as ground truth."""

    facet: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: list[str] = Field(default_factory=list)
    source: FacetSource = "rules"


class ExperienceArtifacts(BaseModel):
    """Links back to durable experiment / reflection artifacts (not category wikis)."""

    experiment_id: str | None = None
    execution_id: str | None = None
    plan_id: str | None = None
    reflection_id: str | None = None
    git_commit: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] = Field(default_factory=dict)


class ExperienceRecord(BaseModel):
    """One cross-competition research episode.

    Retrieval decides what patterns matter; this record does not hardcode
    prompt / HP / paper category stores.
    """

    id: str
    source_competition: str
    goal: str = ""
    hypothesis: str = ""
    hypothesis_id: str | None = None
    action: str = ""
    result: str = ""
    outcome: ExperienceOutcome = "fail"
    artifacts: ExperienceArtifacts = Field(default_factory=ExperienceArtifacts)
    facets: list[ExperienceFacet] = Field(default_factory=list)
    idempotency_key: str
    created_at: datetime
    updated_at: datetime

    @field_validator("facets", mode="before")
    @classmethod
    def _coerce_legacy_tags(cls, value: Any) -> Any:
        """Accept legacy flat string tags from older rows."""
        if not isinstance(value, list):
            return value
        out: list[Any] = []
        for item in value:
            if isinstance(item, str):
                out.append(
                    {
                        "facet": item,
                        "confidence": 0.5,
                        "evidence": [],
                        "source": "legacy",
                    }
                )
            else:
                out.append(item)
        return out

    def facet_names(self) -> list[str]:
        return [f.facet for f in self.facets]
