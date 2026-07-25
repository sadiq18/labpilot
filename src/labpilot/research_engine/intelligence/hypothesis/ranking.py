"""Deterministic ranking SoR for Hypothesis Assistant (§10.4 / §2.4).

LLM may draft text fields only — never replace this score as source of truth.
"""

from __future__ import annotations

from labpilot.research_engine.intelligence.hypothesis.models import HypothesisCandidate
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
)

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


def score_candidate(candidate: HypothesisCandidate) -> float:
    """Explicit reproducible score — stable sort key for top-N."""
    gain = _GAIN_VALUE.get(candidate.expected_impact, 0.35)
    runtime, gpu = _EFFORT_COST.get(candidate.implementation_effort, (0.4, 0.3))
    diversity = min(0.1, 0.02 * len(candidate.evidence))
    failure_bonus = min(0.1, 0.05 * len(candidate.avoids_failure_ids))
    kind_bonus = {
        "pipeline_diff": 0.05,
        "failure_fix": 0.04,
        "transfer": 0.03,
        "technique": 0.0,
        "belief": 0.02,
    }.get(str(candidate.kind), 0.0)
    return (
        gain * _W_GAIN
        + candidate.confidence * _W_CONF
        - runtime * _W_RUNTIME
        - gpu * _W_GPU
        + diversity
        + failure_bonus
        + kind_bonus
        + 0.01 * candidate.score_hint
    )


def rank_candidates(
    candidates: list[HypothesisCandidate], *, limit: int = 10
) -> list[tuple[float, HypothesisCandidate]]:
    """Return top ``limit`` candidates as ``(score, candidate)`` descending."""
    scored = [(score_candidate(candidate), candidate) for candidate in candidates]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].title.lower(),
            item[1].key,
        )
    )
    return scored[: max(0, limit)]
