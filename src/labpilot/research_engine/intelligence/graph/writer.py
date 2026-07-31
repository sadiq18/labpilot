"""Write Research Graph edges from an Evidence Card into SQLite."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.models import (
    ClaimEvidenceKind,
    EvidenceCard,
    EvidenceDecision,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import ResearchArtifact, ResearchArtifactType
from labpilot.research_engine.reflection.store import ReflectionStore

logger = logging.getLogger(__name__)


def write_graph_edges_from_card(
    *,
    knowledge_dir: Path,
    competition: str,
    card: EvidenceCard,
) -> dict[str, Any]:
    """Upsert evidenced edges: technique↔claim↔hyp↔execution↔evidence↔beliefs."""
    summary: dict[str, Any] = {"links": 0, "claims": 0, "techniques": 0}
    if not card.id:
        return summary

    artifact_id = f"exp:execution:{card.treatment_experiment}"
    with KnowledgeStore(knowledge_dir, competition) as kstore:
        # Ensure experiment artifact exists for linking.
        existing = kstore.get_artifact(artifact_id)
        if existing is None:
            kstore.upsert_artifact(
                ResearchArtifact(
                    id=artifact_id,
                    type=ResearchArtifactType.EXPERIMENT,
                    source="labpilot",
                    title=f"execution {card.treatment_experiment}",
                    summary=f"Evidence {card.id} decision={card.decision.value}",
                    techniques=list(card.technique_attribution.keys()),
                    confidence=0.6,
                    competition_slug=competition,
                    metadata={
                        "execution_id": card.treatment_experiment,
                        "hypothesis_id": card.hypothesis_id,
                        "evidence_card_id": card.id,
                        "metrics": {
                            "cv_gain": card.observed.cv_gain,
                            "lb_gain": card.observed.lb_gain,
                        },
                    },
                )
            )

        relation = "mentions"
        if card.decision == EvidenceDecision.ACCEPTED:
            relation = "supports"
        elif card.decision == EvidenceDecision.REJECTED:
            relation = "contradicts"

        weight = abs(float(card.observed.cv_gain or 0.0)) + 0.01
        for tech in card.technique_attribution:
            tid = kstore.merge_technique(
                tech,
                category="experiment",
                summary=f"Used in {card.treatment_experiment}",
                evidence=[artifact_id],
            )
            kstore.link_artifact_technique(
                artifact_id, tid, relation=relation, weight=weight
            )
            summary["techniques"] += 1
            summary["links"] += 1

            # technique used_in hypothesis
            if card.hypothesis_id:
                kstore.add_evidence_link(
                    artifact_id=artifact_id,
                    target_kind="hypothesis",
                    target_id=card.hypothesis_id,
                    relation="used_in",
                    weight=weight,
                    metadata={"evidence_card_id": card.id, "technique": tech},
                )
                summary["links"] += 1

        # hyp executed_as execution
        if card.hypothesis_id:
            kstore.add_evidence_link(
                artifact_id=artifact_id,
                target_kind="execution",
                target_id=card.treatment_experiment,
                relation="executed_as",
                weight=1.0,
                metadata={"hypothesis_id": card.hypothesis_id, "evidence_card_id": card.id},
            )
            summary["links"] += 1

        # execution produced evidence card
        kstore.add_evidence_link(
            artifact_id=artifact_id,
            target_kind="evidence_card",
            target_id=card.id,
            relation="produced",
            weight=weight,
            metadata={
                "cv_gain": card.observed.cv_gain,
                "decision": card.decision.value,
            },
        )
        summary["links"] += 1

    # Claims + claim_evidence
    reflection = ReflectionStore(knowledge_dir, competition)
    try:
        for upd in card.claim_updates:
            claim_id = reflection.upsert_claim_by_statement(
                statement=upd.claim,
                technique=upd.technique,
                confidence=max(0.05, min(0.99, 0.5 + upd.confidence_delta)),
                status=(
                    "supported"
                    if upd.evidence == ClaimEvidenceKind.SUPPORT
                    and card.decision == EvidenceDecision.ACCEPTED
                    else "candidate"
                ),
                metadata={"evidence_card_id": card.id},
            )
            rel = (
                "supports"
                if upd.evidence == ClaimEvidenceKind.SUPPORT
                else "contradicts"
                if upd.evidence == ClaimEvidenceKind.CONTRADICT
                else "mentions"
            )
            reflection.link_claim_evidence(
                claim_id=claim_id,
                evidence_id=card.id,
                relation=rel,
                weight=abs(upd.confidence_delta) + 0.01,
            )
            summary["claims"] += 1
            summary["links"] += 1
    except Exception as exc:
        logger.warning("Claim graph write skipped: %s", exc)
    finally:
        reflection.close()

    return summary
