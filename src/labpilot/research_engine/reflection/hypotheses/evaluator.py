"""HypothesisEvaluator — deterministic status + Micro Agent why-text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.research_engine.reflection.critic.critic import CriticAssessment
from labpilot.research_engine.reflection.hypotheses.micro_agent import (
    HypothesisRevisionAgent,
    HypothesisRevisionDraft,
)
from labpilot.research_engine.reflection.hypotheses.outcomes import (
    HypothesisOutcome,
    classify_hypothesis_failure,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import Hypothesis, HypothesisStatus

logger = logging.getLogger(__name__)

#: Verdicts reached by *measurement*. A failed attempt must not overwrite one.
_SETTLED_STATUSES = frozenset({HypothesisStatus.CONFIRMED, HypothesisStatus.REJECTED})

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

    def record_failed_attempt(
        self,
        hypothesis_id: str,
        *,
        failure_reason: str = "",
        failure_kind: str | None = None,
        attempts: int = 1,
        redundant: bool = False,
    ) -> Hypothesis | None:
        """Move a hypothesis out of `testing` after its experiment failed.

        Nothing did this before. A hypothesis leaves `testing` only when an
        evidence card is written, and a failed execution writes no card — so it
        stayed `testing` indefinitely: out of the pool, never retried, never
        retired. Measured on rogii 2026-08-09: one stuck, three historically.

        `RETRYABLE` returns it to `proposed` so the campaign can pick it again;
        `DEAD_END` rejects it with the reason, so it is never picked again. The
        reason is stored on `actual_outcome` because a retired hypothesis whose
        retirement is unexplained is indistinguishable from one that was simply
        never good — and the first is a finding while the second is noise.
        """
        if not hypothesis_id:
            return None
        outcome, why = classify_hypothesis_failure(
            failure_reason=failure_reason,
            failure_kind=failure_kind,
            attempts=attempts,
            redundant=redundant,
        )
        status = (
            HypothesisStatus.REJECTED
            if outcome is HypothesisOutcome.DEAD_END
            else HypothesisStatus.PROPOSED
        )
        try:
            current = self.store.get(hypothesis_id)
            if current is None:
                return None
            # Never demote a settled verdict. A hypothesis already confirmed or
            # rejected by *evidence* outranks a failed attempt: returning a
            # confirmed one to `proposed` would re-queue work that measurement
            # already answered.
            if current.status in _SETTLED_STATUSES:
                return current
            updated = self.store.update_outcome(
                hypothesis_id,
                actual_outcome=why,
                status=status,
                why=why,
            )
            logger.info("Hypothesis %s → %s: %s", hypothesis_id, status, why)
            return updated
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not kill a run
            logger.warning("could not record failed attempt on %s: %s", hypothesis_id, exc)
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
