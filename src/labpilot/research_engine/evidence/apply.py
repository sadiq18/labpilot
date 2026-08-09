"""Apply Evidence Card to beliefs, hypothesis status/confidence, expected vs actual."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.evidence.models import (
    ClaimEvidenceKind,
    EvidenceCard,
    EvidenceDecision,
)
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.store import ReflectionStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

logger = logging.getLogger(__name__)

_STATUS_MAP = {
    EvidenceDecision.ACCEPTED: HypothesisStatus.CONFIRMED,
    EvidenceDecision.REJECTED: HypothesisStatus.REJECTED,
    EvidenceDecision.INCONCLUSIVE: HypothesisStatus.INCONCLUSIVE,
}


def apply_card_to_beliefs(
    *,
    knowledge_dir: Path,
    competition: str,
    card: EvidenceCard,
) -> list[dict]:
    """Step per-technique belief confidence from card claim_updates / attribution."""
    reflection = ReflectionStore(knowledge_dir, competition)
    results: list[dict] = []
    try:
        with KnowledgeStore(knowledge_dir, competition) as kstore:
            for upd in card.claim_updates:
                tech = upd.technique or _tech_from_claim(upd.claim)
                if not tech:
                    continue
                belief_id = f"belief:{competition}:{_slug(tech)}"
                existing = kstore.get_belief(belief_id)
                prior_conf = float(existing["confidence"]) if existing else 0.5
                prior_status = str(existing["status"]) if existing else "suggested"
                delta = float(upd.confidence_delta)
                if upd.evidence == ClaimEvidenceKind.SUPPORT and delta == 0:
                    delta = 0.06
                if upd.evidence == ClaimEvidenceKind.CONTRADICT and delta == 0:
                    delta = -0.08
                # Stability dampens large steps.
                if card.observed.stability.value == "worse":
                    delta *= 0.5
                elif card.observed.stability.value == "improved":
                    delta *= 1.1
                new_conf = max(0.05, min(0.99, prior_conf + delta))
                new_status = _belief_status(new_conf, upd.evidence)
                effect = (
                    "positive"
                    if upd.evidence == ClaimEvidenceKind.SUPPORT
                    else "negative"
                    if upd.evidence == ClaimEvidenceKind.CONTRADICT
                    else "unknown"
                )
                kstore.upsert_belief(
                    belief_id=belief_id,
                    technique=tech,
                    status=new_status,
                    effect=effect,
                    confidence=new_conf,
                    metadata={
                        "last_evidence_card_id": card.id,
                        "last_execution_id": card.treatment_experiment,
                    },
                )
                update_id = reflection.append_belief_update(
                    belief_id=belief_id,
                    prior_confidence=prior_conf,
                    new_confidence=new_conf,
                    prior_status=prior_status,
                    new_status=new_status,
                    reason=f"Evidence card {card.id}: {upd.claim}",
                    execution_id=card.treatment_experiment,
                    evidence_id=card.id,
                    metadata={
                        "claim_evidence": upd.evidence.value,
                        "confidence_delta": delta,
                        "cv_gain": card.observed.cv_gain,
                    },
                )
                results.append(
                    {
                        "belief_id": belief_id,
                        "prior_confidence": prior_conf,
                        "new_confidence": new_conf,
                        "belief_update_id": update_id,
                    }
                )
    except Exception as exc:
        logger.warning("Belief apply from card failed: %s", exc)
    finally:
        reflection.close()
    return results


def apply_card_to_hypothesis(
    *,
    knowledge_dir: Path,
    competition: str,
    card: EvidenceCard,
) -> None:
    """Map card decision → hyp status; record actual_impact / impact_error."""
    if not card.hypothesis_id:
        return
    store = HypothesisStore(knowledge_dir, competition)
    hyp = store.get(card.hypothesis_id)
    if hyp is None:
        return
    status = _STATUS_MAP.get(card.decision, HypothesisStatus.INCONCLUSIVE)
    actual_bits = []
    if card.observed.cv_gain is not None:
        actual_bits.append(f"cv_gain={card.observed.cv_gain:+.6g}")
    if card.observed.lb_gain is not None:
        actual_bits.append(f"lb_gain={card.observed.lb_gain:+.6g}")
    actual_bits.append(f"decision={card.decision.value}")
    if card.impact_error is not None:
        actual_bits.append(f"impact_error={card.impact_error:+.6g}")
    try:
        store.update_outcome(
            card.hypothesis_id,
            actual_outcome="; ".join(actual_bits),
            public_score=None,
            status=status,
            evidence_run_id=card.treatment_experiment,
            why=(
                f"Evidence card {card.id}: {card.decision_summary}. "
                f"attribution={card.technique_attribution}"
            ),
        )
        # Light confidence nudge from decision + |impact_error|.
        updated = store.get(card.hypothesis_id)
        if updated is not None:
            conf = float(updated.confidence)
            if card.decision == EvidenceDecision.ACCEPTED:
                conf = min(0.99, conf + 0.05)
            elif card.decision == EvidenceDecision.REJECTED:
                conf = max(0.05, conf - 0.05)
            if card.impact_error is not None and abs(card.impact_error) > 0.01:
                # Overestimated → shrink confidence slightly.
                if card.expected.cv_gain and card.observed.cv_gain is not None:
                    if abs(card.observed.cv_gain) < abs(card.expected.cv_gain):
                        conf = max(0.05, conf - 0.03)
            # Persist via model_copy + save path used by store internals
            path = store._path_for(updated.id)  # noqa: SLF001
            bumped = updated.model_copy(update={"confidence": conf})
            # Prefer public API if available
            if hasattr(store, "_save"):
                store._save(bumped)  # noqa: SLF001
            else:
                path.write_text(bumped.model_dump_json(indent=2) + "\n")
    except FileNotFoundError:
        logger.warning("Hypothesis %s missing for evidence card apply", card.hypothesis_id)


def _belief_status(confidence: float, evidence: ClaimEvidenceKind) -> str:
    if evidence == ClaimEvidenceKind.CONTRADICT and confidence < 0.35:
        return "rejected"
    if confidence >= 0.85:
        return "established"
    if confidence >= 0.65:
        return "validated"
    return "suggested"


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "-").replace("/", "-").replace("+", "-")[:64]


def _tech_from_claim(claim: str) -> str:
    parts = (claim or "").split()
    return parts[0] if parts else ""
