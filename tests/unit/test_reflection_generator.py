import pytest
from pathlib import Path

from labpilot.baseline.selector import BaselineChoice
from labpilot.config import LLMConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.profiler.tabular import DatasetProfile
from labpilot.reflection.generator import ReflectionGenerator


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


def _generate(generator, profile, baseline, submission):
    return generator.generate(
        run_id="run-123",
        competition="titanic",
        profile=profile,
        baseline=baseline,
        metrics={"cv_accuracy": 0.81},
        submission=submission,
    )


def test_generate_uses_fallback_text_when_no_llm_client(monkeypatch, profile, baseline, submission):
    # `llm_client=None` alone isn't enough to guarantee "no LLM available":
    # the constructor falls back to the real `resolve_llm_client()`, which
    # would pick up genuine ambient credentials from a developer's real
    # `.env` (a normal, supported setup). Mock it explicitly so this test
    # stays hermetic regardless of the environment it runs in.
    monkeypatch.setattr("labpilot.reflection.generator.resolve_llm_client", lambda config: None)
    generator = ReflectionGenerator(LLMConfig(provider="openai", api_key=""), llm_client=None)

    reflection = _generate(generator, profile, baseline, submission)

    assert "LLM generation not available" in reflection
    assert "OPENAI_API_KEY" in reflection
    assert "cv_accuracy: 0.81" in reflection


def test_generate_fallback_mentions_gemini_env_var_for_gemini_provider(
    monkeypatch, profile, baseline, submission
):
    monkeypatch.setattr("labpilot.reflection.generator.resolve_llm_client", lambda config: None)
    generator = ReflectionGenerator(LLMConfig(provider="gemini", api_key=""), llm_client=None)

    reflection = _generate(generator, profile, baseline, submission)

    assert "GEMINI_API_KEY" in reflection


def test_generate_returns_llm_text_when_client_succeeds(profile, baseline, submission):
    fake_client = FakeLLMClient(response="# AI-generated reflection\n\nTry target encoding.")
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    reflection = _generate(generator, profile, baseline, submission)

    assert reflection == "# AI-generated reflection\n\nTry target encoding."
    assert len(fake_client.calls) == 1
    # The reflection system prompt must actually be read and sent — this was
    # previously dead code (reflection_system.md was never opened).
    system_prompt, _user_prompt = fake_client.calls[0]
    assert system_prompt.strip() != ""


def test_generate_falls_back_when_llm_client_raises(monkeypatch, profile, baseline, submission):
    # `generate()` retries transient failures (see `complete_with_fallback`,
    # `max_attempts=3`) before degrading to fallback text; skip the real
    # backoff sleeps so this stays a fast unit test.
    monkeypatch.setattr("labpilot.llm.client.time.sleep", lambda *_args: None)
    fake_client = FakeLLMClient(error=RuntimeError("network error"))
    generator = ReflectionGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    reflection = _generate(generator, profile, baseline, submission)

    assert "LLM generation not available" in reflection
    assert len(fake_client.calls) == 3


def test_save_appends_submission_links_footer(tmp_path: Path, submission: SubmissionResult):
    generator = ReflectionGenerator(LLMConfig(provider="openai", api_key=""), llm_client=None)
    submission = submission.model_copy(
        update={
            "submissions_url": "https://www.kaggle.com/competitions/titanic/submissions",
            "kernel_url": "https://www.kaggle.com/code/user/kernel/versions/1",
            "submission_mode": "kernel",
        }
    )

    path = generator.save(tmp_path, "# Reflection\n\nBody text.", submission=submission)
    content = path.read_text()

    assert "## Submission links" in content
    assert "competitions/titanic/submissions" in content
    assert "code/user/kernel" in content
