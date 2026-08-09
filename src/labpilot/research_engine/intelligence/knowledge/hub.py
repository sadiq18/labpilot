"""Knowledge Extraction hub — the only writer of merged units and beliefs.

``KnowledgeHub.ingest`` is the explicit API contract the orchestrator calls once
per run: standardized artifacts in, merged Layer-3 units plus Layer-4 beliefs
out. Analyzers keep owning their Layer-1/Layer-2 writes; nothing may skip this
hub to write a belief (design §7 review note, §12.4).

Belief status policy (locked by design §12.4):

- external-only evidence  → ``suggested``
- local experiment evidence present → ``testing``
- ``validated`` / ``established`` are **never** written here.
"""

from __future__ import annotations

import logging

from labpilot.research_engine.intelligence.knowledge.extractor import (
    DEFAULT_ENTITY_TYPES,
    KnowledgeExtractor,
)
from labpilot.research_engine.intelligence.knowledge.merger import (
    ConceptCluster,
    KnowledgeMerger,
)
from labpilot.research_engine.intelligence.knowledge.models import (
    BeliefConfidence,
    BeliefStatus,
    EntityType,
    IngestResult,
    KnowledgeUnit,
    TechniqueBelief,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore, entity_id
from labpilot.research_engine.intelligence.models import ResearchArtifact
from labpilot.research_engine.shared.labels import is_record_reference

logger = logging.getLogger("labpilot.research_engine.intelligence.knowledge.hub")

_MAX_CONFIDENCE = 0.95
HUB_VERSION = "1"


class KnowledgeHub:
    """Consume ``ResearchArtifact`` batches into merged knowledge + beliefs."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        llm_client: object | None = None,
        extractor: KnowledgeExtractor | None = None,
        merger: KnowledgeMerger | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or KnowledgeExtractor()
        self.merger = merger or KnowledgeMerger(llm_client=llm_client)

    @staticmethod
    def ingestion_signature(
        *,
        entity_types: tuple[EntityType, ...] = DEFAULT_ENTITY_TYPES,
        write_beliefs: bool = True,
    ) -> str:
        """Versioned processing identity used to detect stale Hub receipts."""
        entities = ",".join(sorted(str(entity_type) for entity_type in entity_types))
        return f"knowledge-hub:{HUB_VERSION};entities={entities};beliefs={int(write_beliefs)}"

    def pending_artifacts(
        self,
        artifacts: list[ResearchArtifact],
        *,
        entity_types: tuple[EntityType, ...] = DEFAULT_ENTITY_TYPES,
        write_beliefs: bool = True,
    ) -> list[ResearchArtifact]:
        """Return new, changed, or previously failed/skipped artifacts."""
        signature = self.ingestion_signature(
            entity_types=entity_types,
            write_beliefs=write_beliefs,
        )
        return self.store.pending_artifacts(artifacts, signature=signature)

    def ingest(
        self,
        artifacts: list[ResearchArtifact],
        *,
        entity_types: tuple[EntityType, ...] = DEFAULT_ENTITY_TYPES,
        write_beliefs: bool = True,
    ) -> IngestResult:
        result = IngestResult()
        if not artifacts:
            result.notes.append("knowledge hub: no artifacts to ingest.")
            return result

        signature = self.ingestion_signature(
            entity_types=entity_types,
            write_beliefs=write_beliefs,
        )
        candidates = self.extractor.extract(artifacts, entity_types=entity_types)
        if not candidates:
            marked = self.store.mark_artifacts_ingested(
                artifacts,
                signature=signature,
            )
            result.notes.append(
                "knowledge hub: no concept candidates found in "
                f"{len(artifacts)} artifact(s); marked {marked} stored artifact(s) ingested."
            )
            return result

        clusters = self.merger.merge(candidates)
        # `hyp:H-010` is a pointer to a hypothesis, not something anyone can
        # test. `KnowledgeStore.merge_technique` refuses these — and its error
        # tells the caller to filter first, which no caller did, so `research
        # ingest` died on the first one it met.
        dropped = [c for c in clusters if is_record_reference(c.canonical)]
        clusters = [c for c in clusters if not is_record_reference(c.canonical)]
        if dropped:
            result.notes.append(
                "knowledge hub: dropped "
                f"{len(dropped)} record reference(s) mistaken for concepts "
                f"({', '.join(sorted(c.canonical for c in dropped)[:5])})."
            )
        for cluster in clusters:
            unit = self._persist_unit(cluster)
            result.units.append(unit)
            if write_beliefs:
                result.beliefs.append(self._persist_belief(unit))

        by_source: dict[str, int] = {}
        for cluster in clusters:
            by_source[cluster.normalized_by] = by_source.get(cluster.normalized_by, 0) + 1
        result.notes.append(
            f"knowledge hub: {len(candidates)} mention(s) merged into "
            f"{len(result.units)} unit(s) ({_fmt_counts(by_source)})."
        )
        if write_beliefs:
            suggested = sum(1 for b in result.beliefs if b.status is BeliefStatus.SUGGESTED)
            testing = sum(1 for b in result.beliefs if b.status is BeliefStatus.TESTING)
            result.notes.append(
                f"knowledge hub: beliefs suggested={suggested}, testing={testing} "
                "(never validated/established)."
            )
        marked = self.store.mark_artifacts_ingested(
            artifacts,
            signature=signature,
        )
        result.notes.append(f"knowledge hub: marked {marked} stored artifact(s) ingested.")
        return result

    def _persist_unit(self, cluster: ConceptCluster) -> KnowledgeUnit:
        evidence_ids = [ref.artifact_id for ref in cluster.evidence]
        # Join tables key off research_artifacts, so only persisted artifacts
        # can be linked; unpersisted ones still count toward the unit card.
        linkable = self.store.existing_artifact_ids(evidence_ids)
        unit = KnowledgeUnit(
            id=entity_id(cluster.entity_type, cluster.canonical),
            entity_type=cluster.entity_type,
            name=cluster.canonical,
            aliases=cluster.aliases,
            category=cluster.category,
            confidence=_unit_confidence(cluster),
            evidence=cluster.evidence,
            competition_slug=self.store.competition,
            normalized_by=cluster.normalized_by,
            metadata={
                "mention_count": cluster.mention_count,
                "alias_count": len(cluster.aliases),
                "linked_evidence": sorted(linkable),
            },
        )
        self.store.merge_entity(
            str(unit.entity_type),
            unit.name,
            category=unit.category,
            confidence=unit.confidence,
            evidence=sorted(linkable),
            relation="supports",
            metadata={
                "aliases": unit.aliases,
                "normalized_by": unit.normalized_by,
                "external_evidence": unit.external_evidence,
                "local_evidence": unit.local_evidence,
            },
        )
        self.store.write_knowledge_unit(
            str(unit.entity_type), unit.id, unit.model_dump_json(indent=2)
        )
        return unit

    def _persist_belief(self, unit: KnowledgeUnit) -> TechniqueBelief:
        external = unit.external_evidence
        local = unit.local_evidence
        status = BeliefStatus.TESTING if local else BeliefStatus.SUGGESTED
        confidence = BeliefConfidence(
            external=min(_MAX_CONFIDENCE, 0.2 + 0.15 * len(external)) if external else 0.0,
            local=min(_MAX_CONFIDENCE, 0.2 + 0.15 * len(local)) if local else 0.0,
        )
        belief = TechniqueBelief(
            id=f"belief_{unit.id}",
            knowledge_unit_id=unit.id,
            entity_type=unit.entity_type,
            technique=unit.name,
            status=status,
            confidence=confidence,
            external_evidence=external,
            local_evidence=local,
            competition_slug=unit.competition_slug,
            metadata={"normalized_by": unit.normalized_by, "aliases": unit.aliases},
        )
        self.store.upsert_belief(
            belief_id=belief.id,
            technique=belief.technique,
            status=str(belief.status),
            effect=belief.effect,
            confidence=max(confidence.external, confidence.local),
            metadata={
                "entity_type": str(belief.entity_type),
                "knowledge_unit_id": belief.knowledge_unit_id,
                "confidence": confidence.model_dump(),
                "external_evidence": external,
                "local_evidence": local,
                **belief.metadata,
            },
        )
        return belief


def _unit_confidence(cluster: ConceptCluster) -> float:
    """More independent sources and more source kinds raise confidence."""
    kinds = {ref.artifact_type for ref in cluster.evidence if ref.artifact_type}
    score = 0.3 + 0.1 * len(cluster.evidence) + 0.1 * max(0, len(kinds) - 1)
    return round(min(_MAX_CONFIDENCE, score), 3)


def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
