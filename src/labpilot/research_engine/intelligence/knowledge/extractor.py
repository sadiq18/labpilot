"""Knowledge Extraction hub input stage — artifacts to concept candidates.

Every source flows through here; the extractor only reads already-extracted
``ResearchArtifact`` fields (no network, no LLM). Which artifact field feeds an
entity type is table-driven so enabling datasets / architectures / tasks later
is configuration, not new plumbing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.knowledge.models import EntityType, EvidenceRef
from labpilot.research_engine.intelligence.models import ResearchArtifact

#: Artifact list-fields that carry names for each entity type.
ENTITY_SOURCE_FIELDS: dict[EntityType, tuple[str, ...]] = {
    EntityType.TECHNIQUE: ("techniques",),
    EntityType.DATASET: ("datasets",),
    EntityType.ARCHITECTURE: ("models",),
    EntityType.TASK: (),
}

DEFAULT_ENTITY_TYPES: tuple[EntityType, ...] = (EntityType.TECHNIQUE,)


class ConceptCandidate(BaseModel):
    """One raw concept mention plus the artifact it came from."""

    entity_type: EntityType
    name: str
    evidence: EvidenceRef


class KnowledgeExtractor:
    """Project artifacts into per-entity concept candidates."""

    def extract(
        self,
        artifacts: list[ResearchArtifact],
        *,
        entity_types: tuple[EntityType, ...] = DEFAULT_ENTITY_TYPES,
    ) -> list[ConceptCandidate]:
        candidates: list[ConceptCandidate] = []
        for artifact in artifacts:
            for entity_type in entity_types:
                for field in ENTITY_SOURCE_FIELDS.get(entity_type, ()):
                    for raw in getattr(artifact, field, []) or []:
                        name = str(raw).strip()
                        if not name:
                            continue
                        candidates.append(
                            ConceptCandidate(
                                entity_type=entity_type,
                                name=name,
                                evidence=EvidenceRef(
                                    artifact_id=artifact.id,
                                    artifact_type=str(artifact.type),
                                    source=artifact.source,
                                    relation="mentions",
                                    weight=max(0.1, artifact.confidence),
                                    mention=name,
                                ),
                            )
                        )
        return candidates


class ExtractedBundle(BaseModel):
    """Grouped candidates, kept as a model so callers can log/inspect counts."""

    by_entity: dict[EntityType, list[ConceptCandidate]] = Field(default_factory=dict)

    @classmethod
    def from_candidates(cls, candidates: list[ConceptCandidate]) -> ExtractedBundle:
        grouped: dict[EntityType, list[ConceptCandidate]] = {}
        for candidate in candidates:
            grouped.setdefault(candidate.entity_type, []).append(candidate)
        return cls(by_entity=grouped)
