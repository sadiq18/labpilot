"""Reflection pipeline — library orchestration (not a control-flow agent)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.reflection.beliefs.updater import BeliefUpdater
from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter
from labpilot.research_engine.reflection.critic.critic import CriticAssessment, ExperimentCritic
from labpilot.research_engine.reflection.evidence.extractor import EvidenceExtractor
from labpilot.research_engine.reflection.hypotheses.evaluator import HypothesisEvaluator
from labpilot.research_engine.reflection.lessons.generator import LessonGenerator
from labpilot.research_engine.reflection.recommendation.next_experiment import (
    recommend_next_experiment,
)
from labpilot.research_engine.reflection.synthesis.synthesizer import KnowledgeSynthesizer


def run_reflection(
    knowledge_dir: Path,
    competition: str,
    *,
    execution_id: str | None = None,
    workspace_path: Path | str | None = None,
    plan_id: str | None = None,
    hypothesis_id: str | None = None,
    llm_client: Any | None = None,
    persist: bool = True,
    promote_claims: bool = True,
    verify_auto: bool = True,
    verify: Any | None = None,
) -> dict[str, Any]:
    """Evidence → Micro Agents (critic/lessons/…) → durable SoR writes.

    After critic assessment, a soft verification gate can skip belief/claim
    promotion on ``reject`` without failing the pipeline.
    """
    from labpilot.research_engine.shared.verify_artifact import verify_ai_artifact

    extractor = EvidenceExtractor(knowledge_dir, competition)
    critic = ExperimentCritic(llm_client=llm_client)
    beliefs = BeliefUpdater(knowledge_dir, competition)
    hyps = HypothesisEvaluator(knowledge_dir, competition, llm_client=llm_client)
    lessons = LessonGenerator(knowledge_dir, competition, llm_client=llm_client)
    claims = ClaimPromoter(knowledge_dir, competition)
    synth = KnowledgeSynthesizer(knowledge_dir, competition, llm_client=llm_client)
    try:
        evidence = extractor.extract(
            execution_id=execution_id,
            workspace_path=workspace_path,
            plan_id=plan_id,
            hypothesis_id=hypothesis_id,
            persist=persist,
        )
        hyp_id = hypothesis_id or evidence.get("hypothesis_id")
        assessment = critic.assess(evidence)
        verification = verify_ai_artifact(
            "reflection_assessment",
            {
                "competition": competition,
                "evidence_id": evidence.get("id"),
                "recommendation": getattr(assessment, "recommendation", None),
            },
            auto=verify_auto,
            prompt=verify,
        )
        promote = promote_claims and verification.decision != "reject"
        if verification.decision == "reject":
            belief_result: dict[str, Any] = {
                "skipped": True,
                "reason": "verification_reject",
            }
        else:
            belief_result = beliefs.update_from_critic(assessment, evidence)
        hyp_result = hyps.evaluate(
            assessment,
            hypothesis_id=hyp_id,
            evidence_run_id=evidence.get("execution_id") or evidence.get("experiment_id"),
        )
        lesson = lessons.generate(assessment, evidence)
        claim_rows: list[dict[str, Any]] = []
        if promote:
            claim_rows = claims.promote_eligible(evidence_id=evidence.get("id"))
            contradiction = assessment.contradiction
            if (
                contradiction
                and contradiction.has_contradiction
                and belief_result.get("belief_id")
                and evidence.get("id")
            ):
                from labpilot.research_engine.intelligence.knowledge.store import (
                    KnowledgeStore,
                )

                ks = KnowledgeStore(knowledge_dir, competition)
                try:
                    belief = ks.get_belief(belief_result["belief_id"])
                    if belief:
                        contested = claims.promote_from_belief(
                            belief,
                            evidence_id=evidence["id"],
                            contradicting_evidence_id=evidence["id"],
                        )
                        if contested:
                            claim_rows.append(contested)
                finally:
                    ks.close()

        understanding = synth.current_understanding()
        recommendation = recommend_next_experiment(
            understanding,
            assessment_recommendation=assessment.recommendation,
            llm_client=llm_client,
        )
        return {
            "evidence": evidence,
            "assessment": assessment.model_dump(),
            "belief": belief_result,
            "hypothesis": hyp_result,
            "lesson": lesson,
            "claims": claim_rows,
            "understanding": understanding,
            "recommendation": recommendation,
            "verification": verification.model_dump(),
            "needs_review": verification.decision == "spot_check",
        }
    finally:
        extractor.close()
        beliefs.close()
        lessons.close()
        claims.close()
        synth.close()
