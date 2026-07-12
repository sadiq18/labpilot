import pytest

from labpilot.brief.generator import BriefGenerator
from labpilot.competition.models import CompetitionSpec, MetricSpec, ProblemType
from labpilot.config import LLMConfig
from labpilot.profiler.tabular import DatasetProfile


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
def competition() -> CompetitionSpec:
    return CompetitionSpec(
        slug="titanic",
        title="Titanic - Machine Learning from Disaster",
        evaluation_metric=MetricSpec(name="accuracy", direction="maximize"),
        problem_type=ProblemType.TABULAR_CLASSIFICATION,
    )


@pytest.fixture
def profile() -> DatasetProfile:
    return DatasetProfile(competition="titanic", row_count=891, column_count=12)


def test_generate_uses_fallback_text_when_no_llm_client(competition, profile):
    generator = BriefGenerator(LLMConfig(provider="openai", api_key=""), llm_client=None)

    brief = generator.generate(competition, profile)

    assert "LLM generation not available" in brief
    assert "OPENAI_API_KEY" in brief
    assert competition.slug in brief


def test_generate_fallback_mentions_gemini_env_var_for_gemini_provider(competition, profile):
    generator = BriefGenerator(LLMConfig(provider="gemini", api_key=""), llm_client=None)

    brief = generator.generate(competition, profile)

    assert "GEMINI_API_KEY" in brief


def test_generate_returns_llm_text_when_client_succeeds(competition, profile):
    fake_client = FakeLLMClient(response="# AI-generated brief\n\nDo feature engineering.")
    generator = BriefGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    brief = generator.generate(competition, profile)

    assert brief.startswith("## Competition Context")
    assert "# AI-generated brief" in brief
    assert "Do feature engineering." in brief
    assert len(fake_client.calls) == 1


def test_generate_falls_back_when_llm_client_raises(competition, profile):
    fake_client = FakeLLMClient(error=RuntimeError("rate limited"))
    generator = BriefGenerator(
        LLMConfig(provider="openai", api_key="sk-test"), llm_client=fake_client
    )

    brief = generator.generate(competition, profile)

    # Must degrade to the fallback template text rather than propagate.
    assert "LLM generation not available" in brief
    assert competition.slug in brief


def test_generator_defaults_to_create_llm_client_when_not_provided(competition, profile):
    # No `llm_client` argument at all, and no API key configured -> the
    # constructor's own `create_llm_client` call must resolve to `None`
    # rather than raising, so this behaves the same as passing `None`
    # explicitly.
    generator = BriefGenerator(LLMConfig(provider="openai", api_key=""))

    assert generator.llm_client is None
    brief = generator.generate(competition, profile)
    assert "LLM generation not available" in brief
