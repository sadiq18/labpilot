"""HypothesisEvaluator — deterministic status + Micro Agent why-text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import Hypothesis, HypothesisStatus
from labpilot.research_engine.reflection.critic.critic import CriticAssessment
from labpilot.research_engine.reflection.hypotheses.micro_agent import (
    HypothesisRevisionAgent,
    HypothesisRevisionDraft,
)

_OUTCOME_MAP = {
    "confirmed": HypothesisStatus.CONFIRMED,
    "rejected": HypothesisStatus.REJECTED,
    "inconclusive": HypothesisStatus.INCONCLUSIVE,
    "partial": HypothesisStatus.INCONCLUSIVE,
}


class HypothesisEvaluator:
    """Mutate hypothesis file SoR from critic assessment + revision agent."""

    def __init__(
        self, knowledge_dir: Path, competition: str, *, llm_client: Any | None = None
    ) -> None:
        self.store = HypothesisStore(Path(knowledge_dir), competition)
        self._revision = HypothesisRevisionAgent(llm_client=llm_client)

    def mark_testing(self, hypothesis_id: str) -> Hypothesis | None:
        if not hypothesis_id:
            return None
        try:
            return self.store.mark_testing_if_proposed(hypothesis_id)
        except FileNotFoundError:
            return None

    def evaluate(
        self,
        assessment: CriticAssessment,
        *,
        hypothesis_id: str | None,
        evidence_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not hypothesis_id:
            return None
        hypothesis = self.store.get(hypothesis_id)
        if hypothesis is None:
            return None

        draft = run_or_none(
            self._revision,
            StructuredContext(
                competition=hypothesis.competition,
                data={
                    "hypothesis_outcome": assessment.hypothesis_outcome,
                    "likely_cause": assessment.likely_cause,
                    "prediction": hypothesis.prediction,
                    "strength_hint": assessment.confidence_label,
                },
            ),
        )
        if draft is None:
            draft = HypothesisRevisionDraft(
                outcome=assessment.hypothesis_outcome or "inconclusive",
                why=assessment.likely_cause or "LLM revision unavailable.",
                revised_prediction=hypothesis.prediction,
                next_checks=["Re-run reflection with a reachable LLM."],
            )
        elif not isinstance(draft, HypothesisRevisionDraft):
            draft = HypothesisRevisionDraft.model_validate(draft.model_dump())
        status = _OUTCOME_MAP.get(draft.outcome, HypothesisStatus.INCONCLUSIVE)
        updated = self.store.update_status(
            hypothesis_id,
            status,
            evidence_run_id=evidence_run_id,
            why=draft.why,
        )
        return {
            "hypothesis_id": hypothesis_id,
            "status": updated.status.value,
            "why": draft.why,
            "revised_prediction": draft.revised_prediction,
            "next_checks": draft.next_checks,
            "generated_by": "llm" if self._revision.last_used_llm else "template_fallback",
        }
