"""Plan 3 — Micro Agents scaffold.

All tests run with **no API key and no network**: the default ``rule_engine``
path must return valid typed artifacts, and the optional LLM path is exercised
only with in-process fakes.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from pydantic import BaseModel

from labpilot.common.micro_agents import (
    BaseMicroAgent,
    MicroAgent,
    StructuredContext,
    coerce_str_list,
)
from labpilot.research_engine.execution import micro_agents as exec_agents
from labpilot.research_engine.execution.micro_agents.reflection_generator import (
    ReflectionDraft,
    ReflectionGeneratorAgent,
)
from labpilot.research_engine.intelligence import micro_agents as intel_agents
from labpilot.research_engine.intelligence.micro_agents.artifacts import (
    ConceptNormalization,
    ExperimentReview,
    ForumExtract,
    HypothesisDraft,
    PaperExtract,
    RepoExtract,
)


class _StaticClient:
    """Fake LLM client returning a fixed JSON string (never touches network)."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.payload


class _BoomClient:
    def complete(self, system: str, user: str) -> str:
        raise RuntimeError("simulated LLM outage")


INTEL_NAMES = [
    "PaperAnalyzerAgent",
    "RepositoryAnalyzerAgent",
    "ForumAnalyzerAgent",
    "HypothesisGeneratorAgent",
    "ConceptNormalizerAgent",
    "ExperimentReviewerAgent",
]


def test_available_agents_registered() -> None:
    assert intel_agents.available_agents() == sorted(INTEL_NAMES)
    assert exec_agents.available_agents() == ["ReflectionGeneratorAgent"]


@pytest.mark.parametrize("name", INTEL_NAMES)
def test_intel_agent_rule_engine_returns_valid_model(name: str) -> None:
    agent = intel_agents.get_agent(name)  # llm_client=None -> deterministic
    assert isinstance(agent, MicroAgent)
    assert not agent.uses_llm
    result = agent.run(StructuredContext(question="q", text="body"))
    assert isinstance(result, BaseModel)
    assert isinstance(result, agent.output_model)


def test_build_agents_shares_disabled_llm() -> None:
    agents = intel_agents.build_agents()
    assert set(agents) == set(INTEL_NAMES)
    assert all(not a.uses_llm for a in agents.values())


def test_unknown_agent_raises() -> None:
    with pytest.raises(intel_agents.UnknownMicroAgentError):
        intel_agents.get_agent("NopeAgent")


def test_paper_agent_rule_engine_echoes_signals() -> None:
    agent = intel_agents.get_agent("PaperAnalyzerAgent")
    ctx = StructuredContext(
        text="…",
        data={"techniques": ["SpecAugment", "EMA"], "models": "ConvNeXt"},
    )
    out = agent.run(ctx)
    assert isinstance(out, PaperExtract)
    assert out.techniques == ["SpecAugment", "EMA"]
    assert out.models == ["ConvNeXt"]  # scalar coerced to a one-item list
    assert out.datasets == []


def test_concept_normalizer_rule_engine_dedupes() -> None:
    agent = intel_agents.get_agent("ConceptNormalizerAgent")
    ctx = StructuredContext(
        items=["SpecAugment", "Time Masking", "SpecAugment", "Frequency Masking"],
        data={"category": "augmentation"},
    )
    out = agent.run(ctx)
    assert isinstance(out, ConceptNormalization)
    assert out.canonical == "SpecAugment"
    assert out.aliases == ["Time Masking", "Frequency Masking"]
    assert out.category == "augmentation"


def test_experiment_reviewer_diagnoses_mismatch() -> None:
    agent = intel_agents.get_agent("ExperimentReviewerAgent")
    ctx = StructuredContext(data={"cv_delta": 0.012, "lb_delta": -0.006, "changes": ["Mixup"]})
    out = agent.run(ctx)
    assert isinstance(out, ExperimentReview)
    assert "mismatch" in out.diagnosis.lower() or "overfit" in out.diagnosis.lower()
    assert out.suggestions == ["Re-examine effect of: Mixup"]


def test_repo_agent_defaults_components_to_techniques() -> None:
    agent = intel_agents.get_agent("RepositoryAnalyzerAgent")
    ctx = StructuredContext(data={"techniques": ["EMA"], "integration_difficulty": "wat"})
    out = agent.run(ctx)
    assert isinstance(out, RepoExtract)
    assert out.components == ["EMA"]
    assert out.integration_difficulty == "unknown"  # invalid value normalized


def test_forum_and_hypothesis_rule_engine() -> None:
    forum = intel_agents.get_agent("ForumAnalyzerAgent").run(
        StructuredContext(data={"discoveries": ["public LB misleading"]})
    )
    assert isinstance(forum, ForumExtract)
    assert forum.discoveries == ["public LB misleading"]

    hyp = intel_agents.get_agent("HypothesisGeneratorAgent").run(
        StructuredContext(question="improve recall", data={"expected_impact": 0.02})
    )
    assert isinstance(hyp, HypothesisDraft)
    assert hyp.observation == "improve recall"
    assert hyp.expected_impact == pytest.approx(0.02)


def test_llm_path_parses_json_into_model() -> None:
    client = _StaticClient('{"techniques": ["EMA"], "claims": ["+1.2% F1"]}')
    agent = intel_agents.get_agent("PaperAnalyzerAgent", llm_client=client)
    assert agent.uses_llm
    out = agent.run(StructuredContext(text="paper"))
    assert isinstance(out, PaperExtract)
    assert out.techniques == ["EMA"]
    assert out.claims == ["+1.2% F1"]
    assert client.calls == 1


def test_llm_failure_soft_falls_back_to_rule_engine() -> None:
    agent = intel_agents.get_agent("PaperAnalyzerAgent", llm_client=_BoomClient())
    out = agent.run(StructuredContext(data={"techniques": ["EMA"]}))
    assert isinstance(out, PaperExtract)
    assert out.techniques == ["EMA"]  # deterministic fallback, no raise


def test_llm_garbage_falls_back_to_rule_engine() -> None:
    agent = intel_agents.get_agent(
        "PaperAnalyzerAgent", llm_client=_StaticClient("not json at all")
    )
    out = agent.run(StructuredContext(data={"techniques": ["Mixup"]}))
    assert isinstance(out, PaperExtract)
    assert out.techniques == ["Mixup"]


def test_reflection_generator_rule_engine_and_llm() -> None:
    agent = exec_agents.get_agent("ReflectionGeneratorAgent")
    assert isinstance(agent, ReflectionGeneratorAgent)
    ctx = StructuredContext(data={"cv_delta": 0.01, "lb_delta": -0.01, "changes": ["EMA"]})
    out = agent.run(ctx)
    assert isinstance(out, ReflectionDraft)
    assert out.next_steps == ["Investigate: EMA"]
    assert "mismatch" in out.likely_cause.lower()

    llm = _StaticClient('{"summary": "s", "likely_cause": "c", "next_steps": ["x"]}')
    out2 = exec_agents.get_agent("ReflectionGeneratorAgent", llm_client=llm).run(
        StructuredContext(text="notes")
    )
    assert out2.summary == "s"
    assert out2.next_steps == ["x"]


AGENT_CLASSES: list[type[BaseMicroAgent]] = [
    intel_agents.PaperAnalyzerAgent,
    intel_agents.RepositoryAnalyzerAgent,
    intel_agents.ForumAnalyzerAgent,
    intel_agents.HypothesisGeneratorAgent,
    intel_agents.ConceptNormalizerAgent,
    intel_agents.ExperimentReviewerAgent,
    ReflectionGeneratorAgent,
]


@pytest.mark.parametrize("cls", AGENT_CLASSES, ids=lambda c: c.name)
def test_each_agent_dir_has_nonempty_skill_md(cls: type[BaseMicroAgent]) -> None:
    skill = Path(inspect.getfile(cls)).parent / "skill.md"
    assert skill.is_file(), f"missing skill.md for {cls.name}"
    assert skill.read_text(encoding="utf-8").strip(), f"empty skill.md for {cls.name}"


def test_coerce_str_list() -> None:
    assert coerce_str_list(None) == []
    assert coerce_str_list("x") == ["x"]
    assert coerce_str_list(["a", "", "b"]) == ["a", "b"]
    assert coerce_str_list((1, 2)) == ["1", "2"]


def test_no_autonomous_agents_package() -> None:
    """Design forbids an autonomous ``agents/`` package (§2.4)."""
    root = Path(__file__).resolve().parents[2] / "src" / "labpilot" / "research_engine"
    assert not (root / "agents").exists()
    assert not (root / "intelligence" / "agents").exists()
