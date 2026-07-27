"""Research Reflection — durable knowledge after experiments.

Evidence → Critic → Beliefs / Hypotheses → Lessons / Claims → Journal.
See ``docs/milestones/research-reflection/``.
"""

from __future__ import annotations

from labpilot.research_engine.reflection.critic import CriticAssessment, ExperimentCritic
from labpilot.research_engine.reflection.evidence import EvidenceExtractor, assess_strength
from labpilot.research_engine.reflection.pipeline import run_reflection
from labpilot.research_engine.reflection.store import ReflectionStore

__all__ = [
    "CriticAssessment",
    "EvidenceExtractor",
    "ExperimentCritic",
    "ReflectionStore",
    "assess_strength",
    "run_reflection",
]
