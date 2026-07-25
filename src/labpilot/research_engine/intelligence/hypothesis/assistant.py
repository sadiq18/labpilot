"""HypothesisAssistant — recommendations only (design §10).

Pipeline: ResearchContext (progressive) → candidates → deterministic rank →
optional LLM text drafts for top-K → persist Suggested M2 hypotheses.
Never executes training or forks runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.common.micro_agents import StructuredContext
from labpilot.experiments.models import HypothesisCreatedBy, HypothesisOrigin
from labpilot.research_engine.intelligence.hypothesis.candidates import generate_candidates
from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisAssistantResult,
    HypothesisCandidate,
    HypothesisRecommendation,
)
from labpilot.research_engine.intelligence.hypothesis.persist import (
    as_generator,
    default_origin,
    load_existing_technique_tags,
    persist_recommendations,
    write_hypotheses_report,
)
from labpilot.research_engine.intelligence.hypothesis.ranking import rank_candidates
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.artifacts import HypothesisDraft
from labpilot.research_engine.intelligence.micro_agents.hypothesis_generator import (
    HypothesisGeneratorAgent,
)
from labpilot.research_engine.intelligence.repositories.models import TransferOpportunity
from labpilot.research_engine.intelligence.retrieval.context_builder import ContextBuilder
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryType,
    ResearchContext,
)


class HypothesisAssistant:
    """Recommend top-N experiments — never executes."""

    def __init__(
        self,
        *,
        llm_client: object | None = None,
        created_by: HypothesisCreatedBy | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.created_by = created_by

    def recommend(
        self,
        *,
        knowledge_dir: Path,
        competition: str,
        context: ResearchContext | None = None,
        question: str = "",
        pipeline: list[str] | None = None,
        transfers: list[TransferOpportunity] | list[dict[str, Any]] | None = None,
        limit: int = 10,
        persist: bool = True,
        write_report: bool = False,
        progressive: bool = True,
    ) -> HypothesisAssistantResult:
        notes: list[str] = []
        research_context = context
        if research_context is None:
            with KnowledgeStore(knowledge_dir, competition) as store:
                research_context = ContextBuilder(
                    store, llm_client=self.llm_client
                ).build(
                    question or "Suggest next experiments",
                    pipeline=pipeline,
                    query_type=QueryType.HYPOTHESIS_GENERATION,
                    competition={"slug": competition},
                    progressive=progressive,
                )
            notes.extend(research_context.notes)

        tried = load_existing_technique_tags(knowledge_dir, competition)
        candidates = generate_candidates(
            research_context,
            transfers=transfers,
            tried_techniques=tried,
        )
        if not candidates:
            notes.append("hypothesis: no candidates generated from ResearchContext.")
            return HypothesisAssistantResult(notes=notes, context=research_context)

        ranked = rank_candidates(candidates, limit=limit)
        created_by = self.created_by or HypothesisCreatedBy.ANALYZE
        recommendations: list[HypothesisRecommendation] = []
        used_llm_any = False
        for rank, (score, candidate) in enumerate(ranked, start=1):
            drafted, used_llm = self._draft(candidate, research_context)
            used_llm_any = used_llm_any or used_llm
            origins = list(candidate.origins) or [HypothesisOrigin.MIXED]
            recommendations.append(
                HypothesisRecommendation(
                    rank=rank,
                    hypothesis_id="",  # filled on persist
                    title=candidate.title,
                    observation=drafted.observation or candidate.observation,
                    reason=drafted.rationale or candidate.reason,
                    prediction=drafted.prediction or candidate.prediction,
                    expected_impact=candidate.expected_impact,
                    confidence=min(
                        1.0, max(0.0, drafted.confidence or candidate.confidence)
                    ),
                    supporting_evidence=list(candidate.evidence),
                    implementation_effort=candidate.implementation_effort,
                    origins=origins,
                    avoids_failure_ids=list(candidate.avoids_failure_ids),
                    score=round(score, 4),
                    created_by=created_by,
                    generator=as_generator(used_llm),
                    origin=default_origin(origins),
                    tags=list(candidate.tags),
                )
            )

        notes.append(
            f"hypothesis: {len(candidates)} candidate(s) → top {len(recommendations)} "
            f"(generator={'llm' if used_llm_any else 'rule_engine'} drafts)."
        )

        if persist:
            recommendations = persist_recommendations(
                recommendations,
                knowledge_dir=knowledge_dir,
                competition=competition,
            )
            notes.append(
                f"hypothesis: persisted {len(recommendations)} Suggested "
                "hypothesis file(s) (status=proposed)."
            )

        if write_report:
            report_path = (
                Path(knowledge_dir) / competition / "research" / "reports" / "hypotheses.json"
            )
            write_hypotheses_report(
                recommendations, path=report_path, notes=notes
            )
            notes.append(f"hypothesis: wrote {report_path}")

        return HypothesisAssistantResult(
            recommendations=recommendations,
            notes=notes,
            context=research_context,
        )

    def _draft(
        self,
        candidate: HypothesisCandidate,
        context: ResearchContext,
    ) -> tuple[HypothesisDraft, bool]:
        """Optional LLM text draft; ranking already fixed. Rule engine always valid."""
        agent = HypothesisGeneratorAgent(llm_client=self.llm_client)
        evidence_text = context.brief or candidate.reason
        result = agent.run(
            StructuredContext(
                question=candidate.title,
                text=evidence_text[:4000],
                data={
                    "observation": candidate.observation,
                    "prediction": candidate.prediction,
                    "rationale": candidate.reason,
                    "expected_impact": _impact_float(candidate),
                    "confidence": candidate.confidence,
                },
            )
        )
        used_llm = bool(getattr(agent, "last_used_llm", False))
        if isinstance(result, HypothesisDraft):
            return result, used_llm
        return (
            HypothesisDraft(
                observation=candidate.observation,
                prediction=candidate.prediction,
                rationale=candidate.reason,
                expected_impact=_impact_float(candidate),
                confidence=candidate.confidence,
            ),
            False,
        )


def _impact_float(candidate: HypothesisCandidate) -> float:
    mapping = {"high": 0.03, "medium": 0.015, "low": 0.005, "unknown": 0.01}
    return mapping.get(str(candidate.expected_impact), 0.01)
