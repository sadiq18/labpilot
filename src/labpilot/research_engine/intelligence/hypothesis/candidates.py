"""Deterministic candidate generation for Hypothesis Assistant (§10.4)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from labpilot.research_engine.shared.experiments.models import HypothesisEvidenceRef, HypothesisOrigin
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

if TYPE_CHECKING:
    from labpilot.research_engine.intelligence.hypothesis.ledger import ExperimentLedger


def generate_candidates(
    context: ResearchContext,
    *,
    transfers: list[TransferOpportunity] | list[dict[str, Any]] | None = None,
    tried_techniques: set[str] | None = None,
    ledger: ExperimentLedger | None = None,
) -> list[HypothesisCandidate]:
    """Build candidates from beliefs/techniques, pipeline-diff, transfers, failures.

    When ``ledger`` is provided, also mint stacked / unused-belief / unused-claim
    candidates from the full competition inventory.
    """
    tried = {normalize_label(item) for item in (tried_techniques or set())}
    if ledger is not None:
        tried |= {normalize_label(x) for x in ledger.techniques_worked}
        tried |= {normalize_label(x) for x in ledger.winning_stack}
    pipeline = {
        normalize_label(item)
        for item in (context.intent.current_pipeline if context.intent else [])
    }
    if ledger is not None:
        pipeline |= {normalize_label(x) for x in ledger.winning_stack}
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

        # Technique / belief-style suggestion (fresh start — lower priority if win exists).
        if key_label not in tried and not (ledger and ledger.is_failed(name)):
            cite = _cite_technique(name, evidence)
            fresh_conf = confidence * (0.55 if ledger and ledger.winning_hypothesis_id else 1.0)
            candidates.append(
                HypothesisCandidate(
                    key=f"technique:{key_label}",
                    kind=HypothesisCandidateKind.TECHNIQUE,
                    title=f"Try {name}",
                    observation=(
                        f"{name} appears in retrieved knowledge for this competition. {cite}"
                    ),
                    reason=_technique_reason(card),
                    prediction=f"Adopting {name} will improve the primary metric. {cite}",
                    technique=name,
                    expected_impact=impact,
                    confidence=fresh_conf,
                    implementation_effort=effort,
                    evidence=evidence,
                    origins=origins or [HypothesisOrigin.MIXED],
                    tags=[name, "technique"],
                    score_hint=fresh_conf,
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

    if ledger is not None:
        candidates.extend(_candidates_from_ledger(ledger, tried=tried))

    return _dedupe_candidates(candidates)


def _candidates_from_ledger(
    ledger: ExperimentLedger,
    *,
    tried: set[str],
) -> list[HypothesisCandidate]:
    """Stack unused techniques/beliefs/claims onto the winning line when present."""
    from labpilot.research_engine.intelligence.hypothesis.ledger import ExperimentLedger as _L

    assert isinstance(ledger, _L)
    out: list[HypothesisCandidate] = []
    parent_id = ledger.winning_hypothesis_id
    stack = list(ledger.winning_stack)
    avoid = {
        (normalize_label(a), normalize_label(b)) for a, b in ledger.avoid_pairs
    } | {(normalize_label(b), normalize_label(a)) for a, b in ledger.avoid_pairs}

    def _blocked(tech: str) -> bool:
        label = normalize_label(tech)
        if not label or label in tried or ledger.is_failed(tech):
            return True
        for parent_tech in stack:
            if (normalize_label(parent_tech), label) in avoid:
                return True
        return False

    # Untried techniques → stacked when parent win exists, else standalone high coverage.
    for name in ledger.techniques_untried:
        if _blocked(name):
            continue
        record = next((t for t in ledger.techniques if t.name == name), None)
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.MIXED,
                ref=aid,
                note=f"technique {name}",
            )
            for aid in (record.artifact_ids if record else [])[:4]
        ]
        cite = _cite_technique(name, evidence)
        if parent_id:
            new_stack = [*stack, name] if name not in stack else list(stack)
            conf = min(0.95, ledger.winning_confidence + 0.08 + (0.05 if ledger.winning_gain else 0.0))
            impact = ExpectedGain.HIGH if ledger.winning_gain > 0 else ExpectedGain.MEDIUM
            out.append(
                HypothesisCandidate(
                    key=f"stacked:{normalize_label(name)}:{parent_id}",
                    kind=HypothesisCandidateKind.STACKED,
                    title=f"On top of {parent_id}, add {name}",
                    observation=(
                        f"Parent {parent_id} worked with stack [{', '.join(stack) or 'baseline'}]. "
                        f"Unused technique {name} remains. {cite}"
                    ),
                    reason=(
                        f"Stack improvement: keep what worked on {parent_id} and merge {name}. {cite}"
                    ),
                    prediction=(
                        f"Adding {name} on top of {parent_id} will further improve the primary metric."
                    ),
                    technique=name,
                    expected_impact=impact,
                    confidence=conf,
                    implementation_effort=EffortEstimate.HOURS_1,
                    evidence=evidence
                    + [
                        HypothesisEvidenceRef(
                            kind=HypothesisOrigin.EXPERIMENT,
                            ref=parent_id,
                            note="winning parent hypothesis",
                        )
                    ],
                    origins=[HypothesisOrigin.EXPERIMENT, HypothesisOrigin.MIXED],
                    tags=[name, "stacked", "improvement", f"fork:{parent_id}"],
                    parent_hypothesis_id=parent_id,
                    technique_stack=new_stack,
                    score_hint=conf + 0.2,
                )
            )
        else:
            out.append(
                HypothesisCandidate(
                    key=f"untried:{normalize_label(name)}",
                    kind=HypothesisCandidateKind.TECHNIQUE,
                    title=f"Try unused technique {name}",
                    observation=f"Technique {name} is in knowledge but never tested. {cite}",
                    reason=f"Full-ledger coverage: unused technique. {cite}",
                    prediction=f"Adopting {name} will improve the primary metric.",
                    technique=name,
                    expected_impact=ExpectedGain.MEDIUM,
                    confidence=0.55,
                    implementation_effort=EffortEstimate.HOURS_1,
                    evidence=evidence,
                    origins=[HypothesisOrigin.MIXED],
                    tags=[name, "technique", "untried"],
                    technique_stack=[name],
                    score_hint=0.5,
                )
            )

    for belief in ledger.beliefs_unused:
        tech = str(belief.get("technique") or "").strip()
        if not tech or _blocked(tech):
            continue
        bid = str(belief.get("id") or tech)
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.MIXED,
                ref=bid,
                note=str(belief.get("summary") or "unused belief"),
            )
        ]
        cite = _cite_technique(tech, evidence)
        parent = parent_id
        new_stack = [*stack, tech] if parent and tech not in stack else ([tech] if not parent else list(stack))
        conf = min(0.95, (ledger.winning_confidence + 0.06) if parent else 0.58)
        out.append(
            HypothesisCandidate(
                key=f"unused_belief:{normalize_label(tech)}",
                kind=HypothesisCandidateKind.UNUSED_BELIEF,
                title=(
                    f"On top of {parent}, test belief {tech}"
                    if parent
                    else f"Test unused belief: {tech}"
                ),
                observation=(
                    f"Belief about {tech} was never part of a prior hypothesis. {cite}"
                ),
                reason=f"Unused belief fuel. {cite}",
                prediction=f"Testing {tech} will improve the primary metric.",
                technique=tech,
                expected_impact=ExpectedGain.MEDIUM,
                confidence=conf,
                implementation_effort=EffortEstimate.HOURS_1,
                evidence=evidence,
                origins=[HypothesisOrigin.MIXED],
                tags=[tech, "unused_belief", "belief"]
                + ([f"fork:{parent}", "stacked"] if parent else []),
                parent_hypothesis_id=parent,
                technique_stack=new_stack,
                score_hint=conf + 0.12,
            )
        )

    for claim in ledger.claims_unused[:20]:
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        short = text[:80]
        label = normalize_label(short)
        if not label or label in tried:
            continue
        # Derive a technique-ish label from claim text.
        tech = short.split(":")[-1].strip()[:60] or short[:40]
        if _blocked(tech):
            continue
        art_id = str(claim.get("artifact_id") or "")
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.MIXED,
                ref=art_id or short,
                note=text[:160],
            )
        ]
        cite = _cite_technique(tech, evidence)
        parent = parent_id
        new_stack = [*stack, tech] if parent and tech not in stack else [tech]
        conf = min(0.92, (ledger.winning_confidence + 0.05) if parent else 0.52)
        out.append(
            HypothesisCandidate(
                key=f"unused_claim:{label}",
                kind=HypothesisCandidateKind.UNUSED_CLAIM,
                title=(
                    f"On top of {parent}, apply claim → {tech}"
                    if parent
                    else f"Test unused claim: {tech}"
                ),
                observation=f"Claim unused in prior hyps: {text[:180]} {cite}",
                reason=f"Unused claim from artifact {art_id}. {cite}",
                prediction=f"Acting on this claim via {tech} will improve the metric.",
                technique=tech,
                expected_impact=ExpectedGain.MEDIUM,
                confidence=conf,
                implementation_effort=EffortEstimate.HOURS_1,
                evidence=evidence,
                origins=[HypothesisOrigin.MIXED],
                tags=[tech, "unused_claim"]
                + ([f"fork:{parent}", "stacked"] if parent else []),
                parent_hypothesis_id=parent,
                technique_stack=new_stack,
                score_hint=conf + 0.1,
            )
        )

    return out


def _cite_technique(name: str, evidence: list[HypothesisEvidenceRef]) -> str:
    if not evidence:
        return f"(technique {name})"
    refs = "; ".join(f"{ref.kind}:{ref.ref}" for ref in evidence[:3])
    return f"(artifact {refs}; technique {name})"


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
