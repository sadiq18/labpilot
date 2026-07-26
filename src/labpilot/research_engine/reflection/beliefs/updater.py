"""BeliefUpdater — deterministic confidence / status mutations + audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.critic.critic import CriticAssessment
from labpilot.research_engine.reflection.store import ReflectionStore

_STEP = {
    "supports": 0.08,
    "contradicts": -0.1,
    "neutral": 0.0,
}


class BeliefUpdater:
    """Update SQLite beliefs from critic output; append ``belief_updates``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self._reflection = ReflectionStore(self.knowledge_dir, competition)
        self._knowledge = KnowledgeStore(self.knowledge_dir, competition)

    def close(self) -> None:
        self._reflection.close()
        self._knowledge.close()

    def update_from_critic(
        self,
        assessment: CriticAssessment,
        evidence: dict[str, Any],
        *,
        technique: str | None = None,
    ) -> dict[str, Any]:
        technique_name = technique or _technique_from_evidence(evidence)
        belief_id = f"belief:{self.competition}:{_slug(technique_name)}"
        existing = self._knowledge.get_belief(belief_id)
        prior_confidence = float(existing["confidence"]) if existing else 0.5
        prior_status = str(existing["status"]) if existing else "suggested"
        prior_effect = str(existing["effect"]) if existing else "unknown"

        delta = _STEP.get(assessment.belief_effect, 0.0)
        delta *= 0.5 + 0.5 * assessment.confidence
        new_confidence = max(0.05, min(0.99, prior_confidence + delta))
        new_effect = _effect_from(assessment.belief_effect, prior_effect)
        new_status = _status_from(new_confidence, assessment.belief_effect)

        self._knowledge.upsert_belief(
            belief_id=belief_id,
            technique=technique_name,
            status=new_status,
            effect=new_effect,
            confidence=new_confidence,
            metadata={
                "last_execution_id": evidence.get("execution_id"),
                "last_evidence_id": evidence.get("id"),
                "critic_summary": assessment.summary,
            },
        )
        update_id = self._reflection.append_belief_update(
            belief_id=belief_id,
            prior_confidence=prior_confidence,
            new_confidence=new_confidence,
            prior_status=prior_status,
            new_status=new_status,
            reason=assessment.likely_cause or assessment.summary,
            execution_id=evidence.get("execution_id"),
            experiment_id=evidence.get("experiment_id"),
            evidence_id=evidence.get("id"),
            metadata={"belief_effect": assessment.belief_effect},
        )
        return {
            "belief_id": belief_id,
            "prior_confidence": prior_confidence,
            "new_confidence": new_confidence,
            "prior_status": prior_status,
            "new_status": new_status,
            "belief_update_id": update_id,
        }


def _technique_from_evidence(evidence: dict[str, Any]) -> str:
    config = evidence.get("config_summary") or {}
    baseline = config.get("baseline_choice") or {}
    if baseline.get("template_name"):
        return str(baseline["template_name"])
    overrides = config.get("training_overrides") or {}
    recipes = overrides.get("feature_recipes") or []
    if recipes:
        return str(recipes[0])
    return "baseline"


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.lower())[:64]


def _effect_from(belief_effect: str, prior: str) -> str:
    if belief_effect == "supports":
        return "positive"
    if belief_effect == "contradicts":
        return "negative"
    return prior if prior != "unknown" else "unknown"


def _status_from(confidence: float, belief_effect: str) -> str:
    if belief_effect == "contradicts" and confidence < 0.35:
        return "rejected"
    if confidence >= 0.85:
        return "established"
    if confidence >= 0.65:
        return "validated"
    return "suggested"
