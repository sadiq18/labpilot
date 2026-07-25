"""Deterministic candidate generation for Hypothesis Assistant (§10.4)."""

from __future__ import annotations

import re
from typing import Any

from labpilot.experiments.models import HypothesisEvidenceRef, HypothesisOrigin
from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisCandidate,
    HypothesisCandidateKind,
)
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
    TransferOpportunity,
)
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.intelligence.retrieval.models import ResearchContext


def generate_candidates(
    context: ResearchContext,
    *,
    transfers: list[TransferOpportunity] | list[dict[str, Any]] | None = None,
    tried_techniques: set[str] | None = None,
) -> list[HypothesisCandidate]:
    """Build candidates from beliefs/techniques, pipeline-diff, transfers, failures."""
    tried = {normalize_label(item) for item in (tried_techniques or set())}
    pipeline = {
        normalize_label(item)
        for item in (context.intent.current_pipeline if context.intent else [])
    }
    candidates: list[HypothesisCandidate] = []

    for card in context.techniques:
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        key_label = normalize_label(name)
        evidence = _evidence_from_card(card)
        origins = _origins_from_evidence(evidence)
        confidence = float(card.get("confidence") or 0.5)
        impact = _impact_from_confidence(confidence, evidence_count=len(evidence))
        effort = _effort_from_card(card)

        # Technique / belief-style suggestion.
        if key_label not in tried:
            candidates.append(
                HypothesisCandidate(
                    key=f"technique:{key_label}",
                    kind=HypothesisCandidateKind.TECHNIQUE,
                    title=f"Try {name}",
                    observation=f"{name} appears in retrieved knowledge for this competition.",
                    reason=_technique_reason(card),
                    prediction=f"Adopting {name} will improve the primary metric.",
                    technique=name,
                    expected_impact=impact,
                    confidence=confidence,
                    implementation_effort=effort,
                    evidence=evidence,
                    origins=origins or [HypothesisOrigin.MIXED],
                    tags=[name, "technique"],
                    score_hint=confidence,
                )
            )

        # Pipeline-diff: technique not already in local pipeline.
        if key_label not in pipeline and key_label not in tried:
            candidates.append(
                HypothesisCandidate(
                    key=f"pipeline_diff:{key_label}",
                    kind=HypothesisCandidateKind.PIPELINE_DIFF,
                    title=f"Add {name} to the current pipeline",
                    observation=(
                        f"Current pipeline is missing {name}, which has supporting evidence."
                    ),
                    reason=(
                        "Pipeline-diff: "
                        f"{name} is present in domain knowledge but not in "
                        f"[{_pipeline_label(context)}]."
                    ),
                    prediction=(
                        f"Inserting {name} will raise validation score "
                        "vs the current stack."
                    ),
                    technique=name,
                    expected_impact=_bump_impact(impact),
                    confidence=min(0.95, confidence + 0.05),
                    implementation_effort=effort,
                    evidence=evidence,
                    origins=origins or [HypothesisOrigin.MIXED],
                    tags=[name, "pipeline_diff"],
                    score_hint=confidence + 0.1,
                )
            )

    for raw in transfers or []:
        transfer = (
            raw
            if isinstance(raw, TransferOpportunity)
            else TransferOpportunity.model_validate(raw)
        )
        technique = (
            transfer.remote_choice
            or (transfer.deltas[0] if transfer.deltas else "")
            or transfer.summary
        )
        key_label = normalize_label(technique or transfer.repo_id)
        if not key_label or key_label in tried:
            continue
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.REPOSITORY,
                ref=transfer.repo_id,
                note=transfer.summary,
            )
        ]
        candidates.append(
            HypothesisCandidate(
                key=f"transfer:{key_label}",
                kind=HypothesisCandidateKind.TRANSFER,
                title=transfer.hypothesis_hint or f"Transfer: {transfer.summary}",
                observation=transfer.local_baseline or "Local stack differs from a related repo.",
                reason=transfer.summary,
                prediction=(
                    f"Applying {transfer.remote_choice or technique} from "
                    f"{transfer.repo_id} will improve results."
                ),
                technique=str(transfer.remote_choice or technique),
                expected_impact=transfer.expected_gain,
                confidence=0.55,
                implementation_effort=transfer.effort,
                evidence=evidence,
                origins=[HypothesisOrigin.REPOSITORY],
                tags=[technique, "transfer"],
                score_hint=0.55,
                metadata={"repo_id": transfer.repo_id, "deltas": list(transfer.deltas)},
            )
        )

    for failure in context.failures:
        fail_id = str(failure.get("document_id") or failure.get("label") or "")
        label = str(failure.get("label") or fail_id)
        summary = str(failure.get("summary") or failure.get("why") or "")
        technique = _technique_from_failure(label, summary, context)
        key_label = normalize_label(f"fix:{technique or label}")
        technique_label = normalize_label(technique) if technique else ""
        # Skip when this failure's technique was already tried locally (Q5).
        if key_label in tried or (technique_label and technique_label in tried):
            continue
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.EXPERIMENT
                if str(failure.get("kind")) == "failure"
                else HypothesisOrigin.MIXED,
                ref=fail_id or label,
                note=summary[:160],
            )
        ]
        candidates.append(
            HypothesisCandidate(
                key=f"failure_fix:{key_label}",
                kind=HypothesisCandidateKind.FAILURE_FIX,
                title=f"Fix failure: {label}",
                observation=summary or label,
                reason=f"Avoid repeating known failure '{label}'.",
                prediction=(
                    f"A constrained change around {technique or 'this failure mode'} "
                    "will recover the regression."
                ),
                technique=technique,
                expected_impact=ExpectedGain.MEDIUM,
                confidence=0.6,
                implementation_effort=EffortEstimate.MINUTES_20,
                evidence=evidence,
                origins=[HypothesisOrigin.EXPERIMENT],
                avoids_failure_ids=[fail_id] if fail_id else [],
                tags=[technique or label, "failure_fix"],
                score_hint=0.65,
            )
        )

    return _dedupe_candidates(candidates)


def _pipeline_label(context: ResearchContext) -> str:
    if context.intent and context.intent.current_pipeline:
        return ", ".join(context.intent.current_pipeline)
    return "—"


def _evidence_from_card(card: dict[str, Any]) -> list[HypothesisEvidenceRef]:
    refs: list[HypothesisEvidenceRef] = []
    for paper_id in card.get("paper_ids") or []:
        refs.append(HypothesisEvidenceRef(kind=HypothesisOrigin.PAPER, ref=str(paper_id)))
    for exp_id in card.get("experiment_ids") or []:
        refs.append(
            HypothesisEvidenceRef(kind=HypothesisOrigin.EXPERIMENT, ref=str(exp_id))
        )
    for repo_id in card.get("repository_ids") or []:
        refs.append(
            HypothesisEvidenceRef(kind=HypothesisOrigin.REPOSITORY, ref=str(repo_id))
        )
    return refs


def _origins_from_evidence(evidence: list[HypothesisEvidenceRef]) -> list[HypothesisOrigin]:
    kinds: list[HypothesisOrigin] = []
    for ref in evidence:
        try:
            kinds.append(HypothesisOrigin(str(ref.kind)))
        except ValueError:
            continue
    return list(dict.fromkeys(kinds))


def _technique_reason(card: dict[str, Any]) -> str:
    bits = [
        f"confidence={float(card.get('confidence') or 0.5):.2f}",
        f"papers={len(card.get('paper_ids') or [])}",
        f"experiments={len(card.get('experiment_ids') or [])}",
        f"repos={len(card.get('repository_ids') or [])}",
    ]
    issues = str(card.get("known_issues") or "").strip()
    if issues:
        bits.append(f"known_issues={issues[:80]}")
    return "Retrieved technique card: " + "; ".join(bits)


def _impact_from_confidence(confidence: float, *, evidence_count: int) -> ExpectedGain:
    score = confidence + 0.05 * min(evidence_count, 4)
    if score >= 0.75:
        return ExpectedGain.HIGH
    if score >= 0.5:
        return ExpectedGain.MEDIUM
    if score >= 0.3:
        return ExpectedGain.LOW
    return ExpectedGain.UNKNOWN


def _bump_impact(impact: ExpectedGain) -> ExpectedGain:
    order = [
        ExpectedGain.UNKNOWN,
        ExpectedGain.LOW,
        ExpectedGain.MEDIUM,
        ExpectedGain.HIGH,
    ]
    index = order.index(impact) if impact in order else 0
    return order[min(len(order) - 1, index + 1)]


def _effort_from_card(card: dict[str, Any]) -> EffortEstimate:
    # Heuristic: config/aug tricks are cheap; architecture-ish names are heavier.
    name = str(card.get("name") or "").lower()
    if any(token in name for token in ("loss", "mixup", "cutmix", "ema", "swa", "aug")):
        return EffortEstimate.MINUTES_20
    if any(token in name for token in ("efficientnet", "convnext", "transformer", "unet")):
        return EffortEstimate.HOURS_4
    return EffortEstimate.HOURS_1


def _technique_from_failure(
    label: str, summary: str, context: ResearchContext
) -> str:
    haystack = f"{label} {summary}".lower()
    for card in context.techniques:
        name = str(card.get("name") or "")
        if name and name.lower() in haystack:
            return name
    match = re.search(r"known issue:\s*(.+)$", label, flags=re.I)
    if match:
        return match.group(1).strip()
    return ""


def _dedupe_candidates(candidates: list[HypothesisCandidate]) -> list[HypothesisCandidate]:
    best: dict[str, HypothesisCandidate] = {}
    for candidate in candidates:
        existing = best.get(candidate.key)
        if existing is None or candidate.score_hint > existing.score_hint:
            best[candidate.key] = candidate
    return list(best.values())
