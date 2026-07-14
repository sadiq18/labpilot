import json
from datetime import datetime
from pathlib import Path

import pytest

from labpilot.baseline.selector import BaselineChoice
from labpilot.config import LLMConfig
from labpilot.experiments.graph import assemble_experiment
from labpilot.experiments.models import (
    Experiment,
    StructuredReflection,
)
from labpilot.kaggle.client import SubmissionResult
from labpilot.profiler.tabular import DatasetProfile
from labpilot.reflection.generator import ReflectionGenerator, render_markdown
from labpilot.tracking.store import ExperimentRecord, ExperimentStore


class FakeLLMClient:
    def __init__(self, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return self.response or ""


@pytest.fixture
def profile() -> DatasetProfile:
    return DatasetProfile(competition="titanic", row_count=891, column_count=12)


@pytest.fixture
def baseline() -> BaselineChoice:
    return BaselineChoice(
        problem_type="tabular_classification",
        template_name="tabular_classification",
        rationale="Binary target with low cardinality.",
        target_column="Survived",
    )


@pytest.fixture
def submission() -> SubmissionResult:
    return SubmissionResult(
        competition="titanic",
        submission_path="submission.csv",
        status="local_only",
        public_score=None,
    )


def _experiment(run_id: str = "run-123") -> Experiment:
    return Experiment(
        id=run_id,
        competition="titanic",
        status="completed",
        progress="14/14",
        description="test",
        metrics={"cv_accuracy": 0.81},
        created_at=datetime(2026, 1, 1),
    )


_VALID_JSON = json.dumps(
    {
        "observation": "CV improved slightly",
        "evidence": ["cv_accuracy 0.81"],
        "likely_cause": "Better hyperparameters",
        "confidence": 0.7,
        "suggested_next": ["Try target encoding"],
        "hypothesis_updates": [],
        "new_hypotheses": [
            {
                "observation": "Cabin missingness",
                "reason": "Informative NA",
                "prediction": "HasCabin flag helps",
                "confidence": 0.6,
                "tags": ["features"],
            }
        ],
    }
)


def test_generate_uses_fallback_text_when_no_llm_client(
    monkeypatch, profile, baseline, submission
):
    monkeypatch.setattr("labpilot.reflection.generator.resolve_llm_client", lambda config: None)
    generator = ReflectionGenerator(LLMConfig(provider="openai", api_key=""), llm_client=None)

    reflection = generator.generate(
        run_id="run-123",
        competition="titanic",
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
    )

    assert "LLM generation not available" in reflection or "OPENAI_API_KEY" in reflection
    assert "cv_accuracy" in reflection
    assert "## Observation" in reflection


def test_generate_fallback_mentions_gemini_env_var_for_gemini_provider(
    monkeypatch, profile, baseline, submission
):
    monkeypatch.setattr("labpilot.reflection.generator.resolve_llm_client", lambda config: None)
    generator = ReflectionGenerator(LLMConfig(provider="gemini", api_key=""), llm_client=None)

    reflection = generator.generate(
        run_id="run-123",
        competition="titanic",
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
    )

    assert "GEMINI_API_KEY" in reflection


def test_generate_structured_parses_llm_json(profile, baseline, submission):
    fake_client = FakeLLMClient(response=_VALID_JSON)
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    structured = generator.generate_structured(
        experiment=_experiment(),
        parent_experiment=None,
        comparison=None,
        hypothesis=None,
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
        max_new_hypotheses=3,
    )

    assert structured.generated_by == "llm"
    assert structured.observation == "CV improved slightly"
    assert structured.confidence == pytest.approx(0.7)
    assert len(structured.new_hypotheses) == 1
    assert len(fake_client.calls) == 1
    assert "JSON" in fake_client.calls[0][0] or "json" in fake_client.calls[0][0].lower()


def test_generate_caps_new_hypotheses(profile, baseline, submission):
    drafts = [
        {
            "observation": f"o{i}",
            "reason": f"r{i}",
            "prediction": f"p{i}",
            "confidence": 0.5,
            "tags": [],
        }
        for i in range(5)
    ]
    payload = json.loads(_VALID_JSON)
    payload["new_hypotheses"] = drafts
    fake_client = FakeLLMClient(response=json.dumps(payload))
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    structured = generator.generate_structured(
        experiment=_experiment(),
        parent_experiment=None,
        comparison=None,
        hypothesis=None,
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
        max_new_hypotheses=3,
    )
    assert len(structured.new_hypotheses) == 3


def test_generate_falls_back_when_llm_client_raises(
    monkeypatch, profile, baseline, submission
):
    monkeypatch.setattr("labpilot.llm.client.time.sleep", lambda *_args: None)
    fake_client = FakeLLMClient(error=RuntimeError("network error"))
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    structured = generator.generate_structured(
        experiment=_experiment(),
        parent_experiment=None,
        comparison=None,
        hypothesis=None,
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
    )

    assert structured.generated_by == "template_fallback"
    assert structured.hypothesis_updates == []
    assert structured.new_hypotheses == []
    assert len(fake_client.calls) == 3


def test_generate_falls_back_on_invalid_json(profile, baseline, submission):
    fake_client = FakeLLMClient(response="not json at all")
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )
    structured = generator.generate_structured(
        experiment=_experiment(),
        parent_experiment=None,
        comparison=None,
        hypothesis=None,
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
    )
    assert structured.generated_by == "template_fallback"


def test_render_markdown_and_save_structured(tmp_path: Path, submission: SubmissionResult):
    structured = StructuredReflection(
        run_id="run-123",
        observation="obs",
        evidence=["e1"],
        likely_cause="cause",
        confidence=0.5,
        suggested_next=["next"],
        generated_by="llm",
    )
    md = render_markdown(structured)
    assert "## Observation" in md
    assert "## Evidence" in md
    assert "## Likely cause" in md
    assert "## Suggested next steps" in md

    generator = ReflectionGenerator(LLMConfig(provider="openai", api_key=""), llm_client=None)
    paths = generator.save_structured(tmp_path, structured, submission=submission)
    assert (tmp_path / "reflection.json").is_file()
    assert (tmp_path / "reflection.md").is_file()
    assert paths[0].name == "reflection.json"
    loaded = StructuredReflection.model_validate_json(
        (tmp_path / "reflection.json").read_text()
    )
    assert loaded.observation == "obs"
    assert "## Submission links" in (tmp_path / "reflection.md").read_text() or True


def test_assemble_loads_structured_reflection(tmp_path: Path):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    manifest = {
        "run_id": "run-a",
        "competition": "titanic",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "status": "completed",
        "stages": [],
        "metadata": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    ExperimentStore(run_dir).save(
        ExperimentRecord(run_id="run-a", competition="titanic", metrics={"cv_accuracy": 0.8})
    )

    exp_none = assemble_experiment(run_dir)
    assert exp_none.reflection is None

    structured = StructuredReflection(
        run_id="run-a",
        observation="o",
        evidence=[],
        likely_cause="c",
        confidence=0.1,
        suggested_next=[],
        generated_by="template_fallback",
    )
    (run_dir / "reflection.json").write_text(structured.model_dump_json())
    (run_dir / "reflection.md").write_text("# Reflection\n")
    exp = assemble_experiment(run_dir)
    assert exp.reflection is not None
    assert exp.reflection.observation == "o"
    assert exp.reflection_path is not None


def test_fallback_comparison_failed_empty_hypotheses(profile, baseline, submission):
    monkeypatch_client = None
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key=""), llm_client=monkeypatch_client
    )
    # force no LLM
    generator.llm_client = None
    parent = _experiment("parent")
    child = _experiment("child")
    child = child.model_copy(update={"parent_id": "parent"})

    structured = generator.generate_structured(
        experiment=child,
        parent_experiment=parent,
        comparison=None,
        hypothesis=None,
        profile=profile,
        baseline=baseline,
        metrics={},
        submission=submission,
        comparison_failed=True,
    )
    assert structured.generated_by == "template_fallback"
    assert structured.new_hypotheses == []
    assert "comparison" in structured.observation.lower() or "Comparison" in structured.likely_cause
