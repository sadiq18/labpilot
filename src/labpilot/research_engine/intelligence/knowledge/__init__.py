"""Knowledge Store + Extraction hub — layered research tree, SQLite, merge.

- ``store.KnowledgeStore`` — ``research_artifacts`` / merged entity tables /
  join, evidence, and belief tables under ``knowledge.db`` (+ ``extracted/`` cards).
- ``sources.RawStore`` — immutable, versioned Layer-1 blobs under ``raw/``.
- ``hub.KnowledgeHub`` — the only writer of merged units and beliefs (Plan 8).

Storage + merge only: retrieval ranking stays in Plan 9.
"""

from labpilot.research_engine.intelligence.knowledge.extractor import (
    ConceptCandidate,
    KnowledgeExtractor,
)
from labpilot.research_engine.intelligence.knowledge.hub import KnowledgeHub
from labpilot.research_engine.intelligence.knowledge.merger import (
    ConceptCluster,
    KnowledgeMerger,
)
from labpilot.research_engine.intelligence.knowledge.models import (
    BeliefConfidence,
    BeliefStatus,
    EntityType,
    EvidenceRef,
    IngestResult,
    KnowledgeUnit,
    TechniqueBelief,
)
from labpilot.research_engine.intelligence.knowledge.sources import RawStore, RawVersion
from labpilot.research_engine.intelligence.knowledge.store import (
    KnowledgeStore,
    entity_id,
    technique_id,
)

__all__ = [
    "BeliefConfidence",
    "BeliefStatus",
    "ConceptCandidate",
    "ConceptCluster",
    "EntityType",
    "EvidenceRef",
    "IngestResult",
    "KnowledgeExtractor",
    "KnowledgeHub",
    "KnowledgeMerger",
    "KnowledgeStore",
    "KnowledgeUnit",
    "RawStore",
    "RawVersion",
    "TechniqueBelief",
    "entity_id",
    "technique_id",
]
