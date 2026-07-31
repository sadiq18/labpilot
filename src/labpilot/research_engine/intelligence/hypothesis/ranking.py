"""Deterministic ranking SoR for Hypothesis Assistant (§10.4 / §2.4).

LLM may draft text fields only — never replace this score as source of truth.

Preference: combination > stacked > single; diversify top-N; penalize duplicates.
"""

from __future__ import annotations

from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidate
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
)
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

# Design sketch: score = gain*0.5 + confidence*0.2 - runtime*0.1 - gpu*0.2
_W_GAIN = 0.5
_W_CONF = 0.2
_W_RUNTIME = 0.1
_W_GPU = 0.2

_GAIN_VALUE = {
    ExpectedGain.HIGH: 0.85,
    ExpectedGain.MEDIUM: 0.55,
    ExpectedGain.LOW: 0.25,
    ExpectedGain.UNKNOWN: 0.35,
}

# Map effort → (runtime_penalty, gpu_penalty) in [0, 1].
_EFFORT_COST = {
    EffortEstimate.MINUTES_5: (0.05, 0.05),
    EffortEstimate.MINUTES_20: (0.15, 0.10),
    EffortEstimate.HOURS_1: (0.30, 0.20),
    EffortEstimate.HOURS_4: (0.55, 0.40),
    EffortEstimate.DAYS: (0.85, 0.70),
    EffortEstimate.UNKNOWN: (0.40, 0.30),
}

_KIND_BONUS = {
    "pipeline_diff": 0.05,
    "failure_fix": 0.04,
    "transfer": 0.03,
    "technique": 0.0,
    "belief": 0.02,
    "stacked": 0.12,
    "combination": 0.22,
    "ablation": 0.10,
    "unused_belief": 0.08,
    "unused_claim": 0.07,
}


def score_candidate(candidate: HypothesisCandidate) -> float:
    """Explicit reproducible score — stable sort key for top-N."""
    gain = _GAIN_VALUE.get(candidate.expected_impact, 0.35)
    runtime, gpu = _EFFORT_COST.get(candidate.implementation_effort, (0.4, 0.3))
    diversity = min(0.1, 0.02 * len(candidate.evidence))
    failure_bonus = min(0.1, 0.05 * len(candidate.avoids_failure_ids))
    kind_bonus = _KIND_BONUS.get(str(candidate.kind), 0.0)
    parent_bonus = 0.06 if candidate.parent_hypothesis_id else 0.0
    stack_bonus = 0.02 * min(3, max(0, len(candidate.technique_stack) - 1))
    tags_l = {t.lower() for t in candidate.tags}
    if candidate.parent_hypothesis_id and "stacked" in tags_l:
        parent_bonus += 0.04
    if "combination" in tags_l or str(candidate.kind) == "combination":
        parent_bonus += 0.06
    combo = list(candidate.metadata.get("combo_techniques") or [])
    if combo:
        # Mild bonus for multi-technique portfolios (size 2–3).
        stack_bonus += 0.03 * min(3, len(combo))
    return (
        gain * _W_GAIN
        + candidate.confidence * _W_CONF
        - runtime * _W_RUNTIME
        - gpu * _W_GPU
        + diversity
        + failure_bonus
        + kind_bonus
        + parent_bonus
        + stack_bonus
        + 0.01 * candidate.score_hint
    )


def _fingerprint(candidate: HypothesisCandidate) -> str:
    combo = list(candidate.metadata.get("combo_techniques") or [])
    if combo:
        return "combo:" + "+".join(sorted(normalize_label(t) for t in combo))
    tech = normalize_label(candidate.technique or "")
    parent = candidate.parent_hypothesis_id or ""
    if parent and tech:
        return f"stack:{parent}+{tech}"
    return tech or normalize_label(candidate.key)


def _primary_techniques(candidate: HypothesisCandidate) -> set[str]:
    combo = list(candidate.metadata.get("combo_techniques") or [])
    if combo:
        return {normalize_label(t) for t in combo if normalize_label(t)}
    tech = normalize_label(candidate.technique or "")
    return {tech} if tech else set()


def rank_candidates(
    candidates: list[HypothesisCandidate], *, limit: int = 10
) -> list[tuple[float, HypothesisCandidate]]:
    """Return top ``limit`` candidates as ``(score, candidate)`` descending.

    Applies duplicate-fingerprint penalties and soft diversity over techniques so
    the top-N is not dominated by near-identical singles.
    """
    scored = [(score_candidate(candidate), candidate) for candidate in candidates]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].title.lower(),
            item[1].key,
        )
    )
    selected: list[tuple[float, HypothesisCandidate]] = []
    seen_fps: set[str] = set()
    used_techniques: set[str] = set()
    for base_score, candidate in scored:
        fp = _fingerprint(candidate)
        techs = _primary_techniques(candidate)
        penalty = 0.0
        if fp in seen_fps:
            penalty += 0.35
        overlap = techs & used_techniques
        if overlap and str(candidate.kind) != "combination":
            penalty += 0.08 * min(3, len(overlap))
        adjusted = base_score - penalty
        selected.append((adjusted, candidate))
        seen_fps.add(fp)
        used_techniques |= techs
    selected.sort(
        key=lambda item: (
            -item[0],
            item[1].title.lower(),
            item[1].key,
        )
    )
    return selected[: max(0, limit)]
