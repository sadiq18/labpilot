"""Claim promotion stays measurement-first under vocabulary status (M-25 step 2)."""

from __future__ import annotations

from labpilot.research_engine.evidence.models import (
    EvidenceCard,
    EvidenceDecision,
    ObservedOutcomes,
)
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

COMPETITION = "claim-vocab"


def _card(store: EvidenceCardStore, card_id: str, attribution: dict[str, float]) -> None:
    credit = next(iter(attribution.values()), 0.0)
    parent = 194.8
    store.save(
        EvidenceCard(
            id=card_id,
            competition=COMPETITION,
            treatment_experiment="E-1",
            technique_attribution=attribution,
            decision=EvidenceDecision.ACCEPTED,
            maximize=False,
            observed=ObservedOutcomes(
                parent_cv=parent,
                treatment_cv=parent + credit,
                cv_gain=credit,
            ),
        )
    )


def test_novel_measured_technique_promotes_without_store_row(tmp_path) -> None:
    """Measurement licenses the claim — store membership must not be required."""
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-1", {"NovelTrick": -3.83})
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        claim = promoter.promote_from_belief(
            {
                "id": "B-novel",
                "technique": "NovelTrick",
                "effect": "positive",
                "confidence": 0.95,
            }
        )
        assert claim is not None
        assert claim["technique"] == "NovelTrick"
    finally:
        promoter.close()


def test_rejected_status_blocks_promotion_even_when_measured(tmp_path) -> None:
    with KnowledgeStore(tmp_path, COMPETITION) as store:
        tid = store.merge_technique("vit")
        store.set_technique_status(
            tid, "rejected", competition=COMPETITION, reason="test"
        )
    _card(EvidenceCardStore(tmp_path, COMPETITION), "EV-2", {"vit": 5.0})
    promoter = ClaimPromoter(tmp_path, COMPETITION)
    try:
        claim = promoter.promote_from_belief(
            {
                "id": "B-vit",
                "technique": "vit",
                "effect": "positive",
                "confidence": 0.95,
            }
        )
        assert claim is None
    finally:
        promoter.close()
