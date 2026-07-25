"""Knowledge Hub models — Layer 3 units and Layer 4 beliefs (design §7–§8, §12.4).

``KnowledgeUnit`` is the canonical read model for one merged concept; it is
backed by the ``techniques`` (and sibling entity) tables in ``knowledge.db``.
``entity_type`` keeps every hub signature generic so datasets / architectures /
tasks need no structural rework when they are enabled.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    """Kinds of merged knowledge object. Only ``TECHNIQUE`` ships in Plan 8."""

    TECHNIQUE = "technique"
    DATASET = "dataset"
    ARCHITECTURE = "architecture"
    TASK = "task"


class EvidenceRef(BaseModel):
    """One artifact supporting a knowledge unit."""

    artifact_id: str
    artifact_type: str = ""
    source: str = ""
    relation: str = "mentions"
    weight: float = 1.0
    mention: str = ""

    @property
    def is_local(self) -> bool:
        """Local = own experiment history; everything else is external evidence."""
        return self.artifact_type == "experiment" or self.source in {"m2", "local"}


class KnowledgeUnit(BaseModel):
    """One merged concept plus the evidence that produced it."""

    id: str
    entity_type: EntityType = EntityType.TECHNIQUE
    name: str
    aliases: list[str] = Field(default_factory=list)
    category: str = ""
    domain: str = ""
    summary: str = ""
    known_issues: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    competition_slug: str | None = None
    normalized_by: str = "rule_engine"  # rule_engine | alias_seed | llm
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def external_evidence(self) -> list[str]:
        return [ref.artifact_id for ref in self.evidence if not ref.is_local]

    @property
    def local_evidence(self) -> list[str]:
        return [ref.artifact_id for ref in self.evidence if ref.is_local]


class BeliefStatus(StrEnum):
    """Locked lifecycle (design §12.4). The hub never writes past ``TESTING``."""

    SUGGESTED = "suggested"
    TESTING = "testing"
    VALIDATED = "validated"
    ESTABLISHED = "established"
    DEPRECATED = "deprecated"


class BeliefConfidence(BaseModel):
    """Split confidence so external reading never masquerades as local proof."""

    external: float = Field(ge=0.0, le=1.0, default=0.0)
    local: float = Field(ge=0.0, le=1.0, default=0.0)


class TechniqueBelief(BaseModel):
    """Layer 4 belief about one knowledge unit — Suggested for external evidence."""

    id: str
    knowledge_unit_id: str
    entity_type: EntityType = EntityType.TECHNIQUE
    technique: str
    effect: str = "unknown"
    status: BeliefStatus = BeliefStatus.SUGGESTED
    confidence: BeliefConfidence = Field(default_factory=BeliefConfidence)
    external_evidence: list[str] = Field(default_factory=list)
    local_evidence: list[str] = Field(default_factory=list)
    competition_slug: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """What one ``KnowledgeHub.ingest`` call produced."""

    units: list[KnowledgeUnit] = Field(default_factory=list)
    beliefs: list[TechniqueBelief] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
