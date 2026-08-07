"""ExperimentCritic facade — composes Reflection Micro Agents.

Uses RootCause + Confidence + Contradiction agents. Belief/hypothesis enum
math stays deterministic in BeliefUpdater / HypothesisEvaluator.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from labpilot.accessor.common.micro_agents import StructuredContext, run_or_none
from labpilot.research_engine.reflection.confidence import (
    ConfidenceEstimate,
    ConfidenceEstimatorAgent,
)
from labpilot.research_engine.reflection.contradiction import (
    ContradictionDetectorAgent,
    ContradictionReport,
)
from labpilot.research_engine.reflection.critic.micro_agent import (
    ReflectionDraft,
    RootCauseAgent,
)


class CriticAssessment(BaseModel):
    """Structured critic output for BeliefUpdater / HypothesisEvaluator."""

    summary: str = ""
    likely_cause: str = ""
    recommendation: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_label: str = "medium"
    belief_effect: str = "neutral"  # supports | contradicts | neutral
    hypothesis_outcome: str = "inconclusive"  # confirmed|rejected|partial|inconclusive
    next_steps: list[str] = Field(default_factory=list)
    generated_by: str = "rule_engine"
    draft: ReflectionDraft | None = None
    contradiction: ContradictionReport | None = None


class ExperimentCritic:
    """Assess experiment evidence via Reflection Micro Agents."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._root_cause = RootCauseAgent(llm_client=llm_client)
        self._confidence = ConfidenceEstimatorAgent(llm_client=llm_client)
        self._contradiction = ContradictionDetectorAgent(llm_client=llm_client)

    def assess(
        self,
        evidence: dict[str, Any],
        *,
        plan_goal: str = "",
        hypothesis_prediction: str = "",
        prior_belief_effect: str = "",
        prior_claim_ids: list[str] | None = None,
    ) -> CriticAssessment:
        comparison = evidence.get("comparison") or {}
        metrics = evidence.get("metrics") or {}
        strength = str(evidence.get("strength") or "moderate")
        cv_delta = _primary_delta(comparison, metrics)
        changes = _changes_from(evidence, comparison)

        notes = []
        if plan_goal:
            notes.append(f"plan_goal: {plan_goal}")
        if hypothesis_prediction:
            notes.append(f"hypothesis: {hypothesis_prediction}")
        notes.append(f"strength: {strength}")

        signal_data = {
            "cv_delta": cv_delta,
            "lb_delta": metrics.get("lb_delta") or comparison.get("lb_delta"),
            "changes": changes,
            "strength": strength,
            "verdict": comparison.get("verdict"),
            "outcome": comparison.get("outcome"),
            "metrics": metrics,
        }
        context = StructuredContext(
            competition=str(evidence.get("competition_slug") or ""),
            text="\n".join(notes),
            data=signal_data,
        )
        belief_effect, hyp_outcome = _map_outcomes(
            strength=strength,
            cv_delta=_as_float(cv_delta),
            verdict=str(comparison.get("verdict") or ""),
        )

        draft = run_or_none(self._root_cause, context)
        conf = run_or_none(
            self._confidence,
            StructuredContext(
                competition=context.competition,
                data={"strength": strength, "cv_delta": cv_delta},
            ),
        )
        contradiction = run_or_none(
            self._contradiction,
            StructuredContext(
                competition=context.competition,
                data={
                    "belief_effect": belief_effect,
                    "prior_belief_effect": prior_belief_effect,
                    "strength": strength,
                    "prior_claim_ids": prior_claim_ids or [],
                    "evidence_id": evidence.get("id"),
                },
            ),
        )

        if draft is None:
            draft = ReflectionDraft(
                summary=f"strength={strength}",
                likely_cause="LLM critic unavailable; outcomes mapped from evidence strength.",
                next_steps=["Re-run reflection with a reachable LLM."],
            )
        elif not isinstance(draft, ReflectionDraft):
            draft = ReflectionDraft.model_validate(draft.model_dump())

        if conf is None:
            conf = ConfidenceEstimate(
                label="medium",
                score=0.5,
                rationale="LLM confidence estimator unavailable.",
            )
        elif not isinstance(conf, ConfidenceEstimate):
            conf = ConfidenceEstimate.model_validate(conf.model_dump())

        if contradiction is None:
            contradiction = ContradictionReport(
                has_contradiction=False,
                summary="LLM contradiction check unavailable.",
            )
        elif not isinstance(contradiction, ContradictionReport):
            contradiction = ContradictionReport.model_validate(contradiction.model_dump())

        used_llm = (
            self._root_cause.last_used_llm
            or self._confidence.last_used_llm
            or self._contradiction.last_used_llm
        )
        recommendation = draft.next_steps[0] if draft.next_steps else draft.likely_cause
        return CriticAssessment(
            summary=draft.summary,
            likely_cause=draft.likely_cause,
            recommendation=recommendation,
            confidence=conf.score,
            confidence_label=conf.label,
            belief_effect=belief_effect,
            hypothesis_outcome=hyp_outcome,
            next_steps=list(draft.next_steps),
            generated_by="llm" if used_llm else "template_fallback",
            draft=draft,
            contradiction=contradiction,
        )


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _primary_delta(comparison: dict[str, Any], metrics: dict[str, Any]) -> float | None:
    delta = comparison.get("delta")
    if delta is None:
        deltas = comparison.get("metric_deltas") or {}
        primary = comparison.get("primary_metric_key")
        if primary and primary in deltas:
            delta = deltas[primary]
        elif deltas:
            delta = deltas[sorted(deltas.keys())[0]]
    if delta is None:
        for key in ("cv_delta", "primary_delta"):
            if key in metrics:
                delta = metrics[key]
                break
    return _as_float(delta)


def _changes_from(evidence: dict[str, Any], comparison: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for change in comparison.get("changes") or []:
        if isinstance(change, dict):
            changes.append(str(change.get("path") or change.get("key") or change))
        else:
            changes.append(str(change))
    baseline = (evidence.get("config_summary") or {}).get("baseline_choice") or {}
    if baseline.get("template_name"):
        changes.append(f"template:{baseline['template_name']}")
    return changes


def _map_outcomes(
    *,
    strength: str,
    cv_delta: float | None,
    verdict: str,
) -> tuple[str, str]:
    if strength == "rejected" or verdict == "regression":
        return "contradicts", "rejected"
    if strength == "strong" or verdict == "worth_keeping":
        return "supports", "confirmed"
    if strength == "weak" or verdict in {"inconclusive", "not_worth_keeping"}:
        if cv_delta is not None and abs(cv_delta) < 0.002:
            return "neutral", "inconclusive"
        if cv_delta is not None and cv_delta < 0:
            return "contradicts", "rejected"
        return "neutral", "inconclusive"
    if cv_delta is not None and cv_delta > 0.005:
        return "supports", "partial"
    if cv_delta is not None and cv_delta < -0.005:
        return "contradicts", "rejected"
    return "neutral", "inconclusive"
