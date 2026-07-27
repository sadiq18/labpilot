"""Unit tests for Reflection Micro Agents (rule_engine path)."""

from __future__ import annotations

from labpilot.accessor.common.micro_agents import BaseMicroAgent, StructuredContext
from labpilot.research_engine.reflection.confidence import ConfidenceEstimatorAgent
from labpilot.research_engine.reflection.contradiction import ContradictionDetectorAgent
from labpilot.research_engine.reflection.critic import RootCauseAgent
from labpilot.research_engine.reflection.hypotheses import HypothesisRevisionAgent
from labpilot.research_engine.reflection.lessons import LessonGeneratorAgent
from labpilot.research_engine.reflection.recommendation import RecommendationAgent
from labpilot.research_engine.reflection.synthesis import EvidenceSynthesisAgent

_AGENTS: list[type[BaseMicroAgent]] = [
    RootCauseAgent,
    ContradictionDetectorAgent,
    EvidenceSynthesisAgent,
    ConfidenceEstimatorAgent,
    LessonGeneratorAgent,
    HypothesisRevisionAgent,
    RecommendationAgent,
]


def test_all_reflection_agents_rule_engine() -> None:
    ctx = StructuredContext(
        competition="demo",
        data={
            "cv_delta": 0.02,
            "strength": "strong",
            "belief_effect": "contradicts",
            "prior_belief_effect": "positive",
            "evidence_by_strength": {"strong": 1, "rejected": 0},
            "open_hypothesis_ids": ["H-001"],
            "hypothesis_outcome": "confirmed",
            "likely_cause": "metric improved",
            "prediction": "mixup helps",
        },
    )
    for cls in _AGENTS:
        out = cls().run(ctx)
        assert out is not None


def test_recommendation_prefers_open_hypothesis() -> None:
    draft = RecommendationAgent().run(
        StructuredContext(
            data={
                "open_hypothesis_ids": ["H-002"],
                "evidence_by_strength": {},
            }
        )
    )
    assert draft.action == "plan_create"
    assert "H-002" in draft.command


def test_contradiction_and_confidence() -> None:
    contra = ContradictionDetectorAgent().run(
        StructuredContext(
            data={
                "belief_effect": "contradicts",
                "prior_belief_effect": "positive",
                "strength": "rejected",
                "evidence_id": "EE-001",
            }
        )
    )
    assert contra.has_contradiction is True
    conf = ConfidenceEstimatorAgent().run(
        StructuredContext(data={"strength": "strong", "cv_delta": 0.02})
    )
    assert conf.label == "high"
    assert RootCauseAgent().name == "RootCauseAgent"
    assert LessonGeneratorAgent().run(
        StructuredContext(data={"strength": "rejected", "likely_cause": "failed"})
    ).category == "pitfall"
    assert HypothesisRevisionAgent().run(
        StructuredContext(
            data={"hypothesis_outcome": "partial", "likely_cause": "noisy"}
        )
    ).outcome == "partial"
    assert EvidenceSynthesisAgent().run(
        StructuredContext(data={"evidence_by_strength": {"strong": 2}})
    ).summary


def test_no_central_reflection_micro_agents_package() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    assert not (root / "reflection" / "micro_agents").exists()
