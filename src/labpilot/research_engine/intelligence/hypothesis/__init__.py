"""Hypothesis Assistant — top-N experiment recommendations (Plan 10).

Deterministic ranking SoR; optional LLM drafts text only. Never auto-runs.
"""

from labpilot.research_engine.intelligence.hypothesis.assistant import HypothesisAssistant
from labpilot.research_engine.intelligence.hypothesis.candidates import generate_candidates
from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisAssistantResult,
    HypothesisCandidate,
    HypothesisCandidateKind,
    HypothesisRecommendation,
)
from labpilot.research_engine.intelligence.hypothesis.persist import (
    persist_recommendations,
    write_hypotheses_report,
)
from labpilot.research_engine.intelligence.hypothesis.ranking import (
    rank_candidates,
    score_candidate,
)

__all__ = [
    "HypothesisAssistant",
    "HypothesisAssistantResult",
    "HypothesisCandidate",
    "HypothesisCandidateKind",
    "HypothesisRecommendation",
    "generate_candidates",
    "persist_recommendations",
    "rank_candidates",
    "score_candidate",
    "write_hypotheses_report",
]
