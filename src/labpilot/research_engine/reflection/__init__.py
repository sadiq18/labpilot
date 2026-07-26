"""Research Reflection — durable knowledge after experiments.

Evidence → Critic → Beliefs / Hypotheses → Lessons / Claims → Journal.
See ``docs/milestones/research-reflection/``.
"""

from __future__ import annotations

from labpilot.research_engine.reflection.evidence import EvidenceExtractor, assess_strength
from labpilot.research_engine.reflection.store import ReflectionStore

__all__ = ["EvidenceExtractor", "ReflectionStore", "assess_strength"]
