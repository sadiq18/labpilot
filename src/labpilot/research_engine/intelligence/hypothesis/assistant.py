"""HypothesisAssistant — recommendations only (design §10).

Pipeline: ResearchContext (progressive) → candidates → deterministic rank →
optional LLM text drafts for top-K → persist Suggested M2 hypotheses.
Never executes training or forks runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.hypothesis.candidates import generate_candidates
from labpilot.research_engine.intelligence.hypothesis.combo import (
    build_combo_shortlist,
    filter_picks_to_shortlist,
    picks_to_candidates,
)
from labpilot.research_engine.intelligence.hypothesis.ledger import build_experiment_ledger
from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisAssistantResult,
    HypothesisCandidate,
    HypothesisRecommendation,
)
from labpilot.research_engine.intelligence.hypothesis.persist import (
    as_generator,
    default_origin,
    load_existing_technique_tags,
    load_open_hypothesis_tags,
    persist_recommendations,
    write_hypotheses_report,
)
from labpilot.research_engine.intelligence.hypothesis.ranking import rank_candidates
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.artifacts import (
    ComboPortfolioDraft,
    HypothesisDraft,
)
from labpilot.research_engine.intelligence.micro_agents.combo_portfolio import (
    ComboPortfolioAgent,
)
from labpilot.research_engine.intelligence.micro_agents.hypothesis_generator import (
    HypothesisGeneratorAgent,
)
from labpilot.research_engine.intelligence.repositories.models import TransferOpportunity
from labpilot.research_engine.intelligence.retrieval.context_builder import ContextBuilder
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.intelligence.retrieval.models import (
    QueryType,
    ResearchContext,
)
from labpilot.research_engine.shared.experiments.models import HypothesisCreatedBy, HypothesisOrigin

#: Candidate-kind marker tags — not techniques, so they never match a backlog tag.
_KIND_TAGS = frozenset(
    {
        "technique",
        "pipeline_diff",
        "transfer",
        "failure_fix",
        "stacked",
        "improvement",
        "untried",
        "unused_belief",
        "unused_claim",
        "belief",
        "combination",
        "ablation",
    }
)


def _candidate_labels(candidate: HypothesisCandidate) -> set[str]:
    """Normalized technique labels a candidate would be filed under."""
    labels = {
        normalize_label(tag)
        for tag in candidate.tags
        if tag not in _KIND_TAGS and not str(tag).lower().startswith("fork:")
    }
    if candidate.technique:
        labels.add(normalize_label(candidate.technique))
    combo = list(candidate.metadata.get("combo_techniques") or [])
    if combo:
        joined = "+".join(sorted(normalize_label(t) for t in combo))
        labels = {
            joined,
            normalize_label(
                f"{candidate.parent_hypothesis_id or 'root'}+{joined}"
            ),
        }
        if candidate.technique:
            labels.add(normalize_label(candidate.technique))
        return {label for label in labels if label}
    # Stacked candidates are unique per parent+technique, not technique alone.
    if candidate.parent_hypothesis_id and candidate.technique:
        labels = {
            normalize_label(f"{candidate.parent_hypothesis_id}+{candidate.technique}")
        }
    return {label for label in labels if label}


def _resolve_problem_type(knowledge_dir: Path, competition: str) -> str:
    """Best-effort problem type, used to reject cross-modality techniques.

    Falls through to `baseline_choice.json` when the competition spec says
    ``unknown``. Measured on rogii 2026-08-07: `competition.json` carries
    ``unknown`` (it is written before the data is profiled) while
    `baseline_choice.json` — derived from the profile — says
    ``tabular_regression``.

    That mattered: an empty/unknown type makes `filter_incompatible_techniques`
    return early, so the cross-modality filter was **disabled entirely**. `vit`
    was proposed for a tabular regression, executed, and then propagated into
    every subsequent `technique_stack`. The filter was correctly configured all
    along — it was reading the one source of three that did not know.
    """
    from labpilot.research_engine.intelligence.competition.models import CompetitionSpec

    for candidate in (
        knowledge_dir.parent / "competition.json",
        knowledge_dir / competition / "competition.json",
    ):
        try:
            if candidate.is_file():
                spec = CompetitionSpec.model_validate_json(
                    candidate.read_text(encoding="utf-8")
                )
                resolved = str(spec.problem_type or "").strip()
                if resolved and resolved.lower() != "unknown":
                    return resolved
        except Exception:  # noqa: BLE001 — filtering is an optimisation, not a gate
            continue

    for derived in (
        knowledge_dir.parent / "baseline_choice.json",
        knowledge_dir / competition / "baseline_choice.json",
    ):
        try:
            if derived.is_file():
                data = json.loads(derived.read_text(encoding="utf-8"))
                resolved = str(data.get("problem_type") or "").strip()
                if resolved and resolved.lower() != "unknown":
                    return resolved
        except Exception:  # noqa: BLE001
            continue
    return ""


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
        technique_statuses: dict[str, str] = {}
        with KnowledgeStore(knowledge_dir, competition) as store:
            if research_context is None:
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
            technique_statuses = {}
            for row in store.list_techniques():
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                technique_statuses[normalize_label(name)] = str(
                    row.get("status") or "candidate"
                )

        tried = load_existing_technique_tags(knowledge_dir, competition)
        ledger = build_experiment_ledger(knowledge_dir, competition)
        notes.append(
            "hypothesis: ledger "
            f"artifacts={len(ledger.artifacts)} "
            f"untried={len(ledger.techniques_untried)} "
            f"worked={len(ledger.techniques_worked)} "
            f"failed={len(ledger.techniques_failed)} "
            f"unused_beliefs={len(ledger.beliefs_unused)} "
            f"winning={ledger.winning_hypothesis_id or '—'}"
        )
        candidates = generate_candidates(
            research_context,
            transfers=transfers,
            tried_techniques=tried,
            ledger=ledger,
            problem_type=_resolve_problem_type(knowledge_dir, competition),
            technique_statuses=technique_statuses,
        )
        combo_candidates, combo_note = self._combo_candidates(
            ledger, research_context=research_context
        )
        if combo_note:
            notes.append(combo_note)
        if combo_candidates:
            candidates = [*combo_candidates, *candidates]
        try:
            from labpilot.research_engine.intelligence.graph.query import (
                query_techniques,
            )

            graph_hits = {
                str(row["technique"]).lower(): float(row["confidence"])
                for row in query_techniques(
                    knowledge_dir=knowledge_dir,
                    competition=competition,
                    limit=50,
                )
            }
            enriched: list[HypothesisCandidate] = []
            for cand in candidates:
                tech = (cand.technique or "").lower()
                conf = graph_hits.get(tech)
                if conf is None:
                    for member in cand.metadata.get("combo_techniques") or []:
                        conf = graph_hits.get(str(member).lower())
                        if conf is not None:
                            break
                if conf is not None:
                    meta = dict(cand.metadata)
                    meta["graph_confidence"] = conf
                    cand = cand.model_copy(update={"metadata": meta})
                enriched.append(cand)
            candidates = enriched
        except Exception:
            pass
        if not candidates:
            notes.append("hypothesis: no candidates generated from ResearchContext.")
            return HypothesisAssistantResult(notes=notes, context=research_context)

        open_tags = load_open_hypothesis_tags(knowledge_dir, competition)
        fresh = [c for c in candidates if not _candidate_labels(c) & open_tags]
        skipped = len(candidates) - len(fresh)
        if skipped:
            notes.append(
                f"hypothesis: skipped {skipped} candidate(s) already covered by an "
                "open hypothesis."
            )
        if not fresh:
            notes.append("hypothesis: 0 new hypothesis generated (backlog already covers these).")
            return HypothesisAssistantResult(notes=notes, context=research_context)
        candidates = fresh

        ranked = rank_candidates(candidates, limit=limit)
        created_by = self.created_by or HypothesisCreatedBy.ANALYZE
        recommendations: list[HypothesisRecommendation] = []
        used_llm_any = False
        for rank, (score, candidate) in enumerate(ranked, start=1):
            drafted, used_llm = self._draft(candidate, research_context)
            used_llm_any = used_llm_any or used_llm
            origins = list(candidate.origins) or [HypothesisOrigin.MIXED]
            observation = drafted.observation or candidate.observation
            reason = drafted.rationale or candidate.reason
            prediction = drafted.prediction or candidate.prediction
            # Preserve technique/artifact citations even if LLM polishes text.
            if candidate.technique and candidate.technique not in observation:
                observation = f"{observation} (technique {candidate.technique})"
            if candidate.evidence and "artifact" not in reason.lower():
                refs = "; ".join(f"{e.kind}:{e.ref}" for e in candidate.evidence[:3])
                reason = f"{reason} (artifact {refs}; technique {candidate.technique})"
            recommendations.append(
                HypothesisRecommendation(
                    rank=rank,
                    hypothesis_id="",  # filled on persist
                    title=candidate.title,
                    observation=observation,
                    reason=reason,
                    prediction=prediction,
                    expected_impact=candidate.expected_impact,
                    expected_impact_value=(
                        drafted.expected_impact or _impact_float(candidate)
                    ),
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
                    technique=candidate.technique,
                    parent_hypothesis_id=candidate.parent_hypothesis_id,
                    technique_stack=list(candidate.technique_stack),
                    combo_techniques=list(
                        candidate.metadata.get("combo_techniques") or []
                    ),
                    combo_rationale=str(
                        candidate.metadata.get("combo_rationale") or ""
                    ),
                )
            )

        notes.append(
            f"hypothesis: {len(candidates)} candidate(s) → top {len(recommendations)} "
            f"(generator={'llm' if used_llm_any else 'rule_engine'} drafts)."
        )

        new_count = 0
        if persist:
            recommendations = persist_recommendations(
                recommendations,
                knowledge_dir=knowledge_dir,
                competition=competition,
            )
            new_count = len(recommendations)
            notes.append(f"hypothesis: {new_count} new hypothesis generated (status=proposed).")

        if write_report:
            from labpilot.research_engine.intelligence.paths import ResearchPaths

            report_path = ResearchPaths(Path(knowledge_dir), competition).reports_dir / (
                "hypotheses.json"
            )
            write_hypotheses_report(
                recommendations, path=report_path, notes=notes
            )
            notes.append(f"hypothesis: wrote {report_path}")

        return HypothesisAssistantResult(
            recommendations=recommendations,
            new_count=new_count,
            notes=notes,
            context=research_context,
        )

    def _combo_candidates(
        self,
        ledger: object,
        *,
        research_context: ResearchContext,
    ) -> tuple[list[HypothesisCandidate], str]:
        shortlist = build_combo_shortlist(ledger)  # type: ignore[arg-type]
        if not shortlist:
            return [], "hypothesis: combo shortlist empty (need ≥2 untried techniques)."
        agent = ComboPortfolioAgent(llm_client=self.llm_client)
        draft = agent.run(
            StructuredContext(
                competition=str(
                    (research_context.competition or {}).get("slug")
                    or getattr(ledger, "competition", "")
                ),
                text=(research_context.brief or "")[:3000],
                data={
                    "shortlist": shortlist,
                    "limit": 3,
                    "parent_stack": list(getattr(ledger, "winning_stack", []) or []),
                    "parent_metrics": {},
                    "avoid_pairs": [
                        list(pair) for pair in getattr(ledger, "avoid_pairs", []) or []
                    ],
                    "failed": list(getattr(ledger, "techniques_failed", []) or []),
                    "skill_agent_key": "combo_portfolio",
                },
            )
        )
        picks_raw: list[dict[str, Any]] = []
        if isinstance(draft, ComboPortfolioDraft):
            picks_raw = [p.model_dump(mode="json") for p in draft.picks]
        picks = filter_picks_to_shortlist(picks_raw, shortlist)
        if not picks:
            from labpilot.research_engine.intelligence.hypothesis.combo import (
                rule_engine_pick_combos,
            )

            picks = rule_engine_pick_combos(shortlist, limit=3)
        candidates = picks_to_candidates(picks, ledger)  # type: ignore[arg-type]
        source = agent.last_generated_by
        note = (
            f"hypothesis: combo shortlist={len(shortlist)} → "
            f"picks={len(candidates)} ({source})"
        )
        return candidates, note

    def _draft(
        self,
        candidate: HypothesisCandidate,
        context: ResearchContext,
    ) -> tuple[HypothesisDraft, bool]:
        """Optional LLM text draft; ranking already fixed. Rule engine always valid."""
        # Combination text is already curated; skip polishing that drops combo refs.
        if candidate.kind.value == "combination":
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
    meta_val = candidate.metadata.get("expected_impact_value")
    if isinstance(meta_val, (int, float)) and float(meta_val) > 0:
        return float(meta_val)
    mapping = {"high": 0.03, "medium": 0.015, "low": 0.005, "unknown": 0.01}
    return mapping.get(str(candidate.expected_impact), 0.01)
