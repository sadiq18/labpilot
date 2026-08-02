"""ClaimPromoter — promote strong beliefs to research_claims."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.store import ReflectionStore

_MIN_CONFIDENCE = 0.7
_MIN_STRONG_EVIDENCE = 1


class ClaimPromoter:
    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.competition = competition
        self._reflection = ReflectionStore(Path(knowledge_dir), competition)
        self._knowledge = KnowledgeStore(Path(knowledge_dir), competition)

    def close(self) -> None:
        self._reflection.close()
        self._knowledge.close()

    def promote_from_belief(
        self,
        belief: dict[str, Any],
        *,
        evidence_id: str | None = None,
        contradicting_evidence_id: str | None = None,
        needs_review: bool = False,
    ) -> dict[str, Any] | None:
        confidence = float(belief.get("confidence") or 0)
        if confidence < _MIN_CONFIDENCE and not contradicting_evidence_id:
            return None

        strong = [
            e
            for e in self._reflection.list_evidence()
            if e.get("strength") == "strong"
        ]
        if (
            len(strong) < _MIN_STRONG_EVIDENCE
            and not contradicting_evidence_id
            and confidence < 0.85
        ):
            return None

        technique = str(belief.get("technique") or "")
        effect = str(belief.get("effect") or "")
        statement = f"{technique} appears to be {effect} on {self.competition}"
        status = "candidate"
        contradictions: list[str] = []
        if contradicting_evidence_id:
            status = "contested"
            contradictions.append(contradicting_evidence_id)

        meta: dict[str, Any] = {}
        if needs_review:
            meta["needs_review"] = True
        claim = self._reflection.create_claim(
            statement,
            status=status,
            confidence=confidence,
            technique=technique,
            effect=effect,
            promoted_from=belief.get("id"),
            contradictions=contradictions,
            metadata=meta or None,
        )
        link_id = evidence_id or (strong[0]["id"] if strong else None)
        if link_id:
            relation = "contradicts" if contradicting_evidence_id == link_id else "supports"
            self._reflection.link_claim_evidence(
                claim["id"], link_id, relation=relation
            )
        return claim

    def promote_eligible(
        self,
        *,
        evidence_id: str | None = None,
        needs_review: bool = False,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for belief in self._knowledge.list_beliefs():
            if float(belief.get("confidence") or 0) < _MIN_CONFIDENCE:
                continue
            if belief.get("status") == "rejected":
                continue
            claim = self.promote_from_belief(
                belief,
                evidence_id=evidence_id,
                needs_review=needs_review,
            )
            if claim:
                created.append(claim)
        return created
