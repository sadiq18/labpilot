import sys

import pytest

from labpilot.config import LLMConfig
from labpilot.llm import client as llm_client_module
from labpilot.llm.client import (
    GeminiClient,
    OpenAIClient,
    complete_with_fallback,
    create_llm_client,
    resolve_llm_client,
)

pytestmark = pytest.mark.llm


def _config(provider: str = "openai", api_key: str = "", model: str = "") -> LLMConfig:
    return LLMConfig(provider=provider, model=model, temperature=0.3, api_key=api_key)


def test_create_llm_client_returns_none_without_api_key():
    # No key configured -> `None` before ever attempting to import a
    # provider package. This is the common case for most users and must
    # never raise.
    assert create_llm_client(_config(provider="openai", api_key="")) is None


def test_create_llm_client_returns_none_for_unknown_provider():
    assert create_llm_client(_config(provider="not-a-real-provider", api_key="sk-test")) is None


@pytest.mark.parametrize(
    ("provider", "module_name"),
    [("openai", "openai"), ("gemini", "google.genai")],
)
def test_create_llm_client_returns_none_when_package_not_installed(
    monkeypatch, provider, module_name
):
    # Simulate the optional provider package not being installed by making
    # its import raise `ImportError`, the same as it would in an
    # environment where `uv sync --extra llm` was never run.
    monkeypatch.setitem(sys.modules, module_name, None)

    client = create_llm_client(_config(provider=provider, api_key="sk-test"))

    assert client is None


def test_create_llm_client_returns_openai_client_with_key_and_package_available():
    pytest.importorskip("openai")

    client = create_llm_client(_config(provider="openai", api_key="sk-test"))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4o-mini"


def test_create_llm_client_returns_gemini_client_with_key_and_package_available():
    pytest.importorskip("google.genai")

    client = create_llm_client(_config(provider="gemini", api_key="fake-key"))

    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-3.5-flash-lite"


def test_create_llm_client_respects_explicit_model_override():
    pytest.importorskip("openai")

    client = create_llm_client(_config(provider="openai", api_key="sk-test", model="gpt-4.1"))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4.1"


def test_default_model_by_provider_matches_client_registry():
    assert set(llm_client_module.DEFAULT_MODEL_BY_PROVIDER) == {"openai", "gemini", "ollama"}


def test_resolve_llm_client_falls_back_to_gemini_when_openai_unavailable(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setitem(sys.modules, "openai", None)

    client = resolve_llm_client(
        _config(provider="openai", api_key="sk-test"),
        alternate_keys={"openai": "sk-test", "gemini": "gemini-key"},
    )

    assert isinstance(client, GeminiClient)


def test_resolve_llm_client_keeps_explicit_model_for_the_primary_provider(monkeypatch):
    # Regression test: `resolve_llm_client()` must not silently discard a
    # user's explicit `model` (e.g. via `LABPILOT_LLM_MODEL`) in favor of
    # the hardcoded `DEFAULT_MODEL_BY_PROVIDER` default when the resolved
    # provider *is* the one the user actually configured — different
    # models can have very different (and independently exhausted) quota.
    pytest.importorskip("google.genai")

    client = resolve_llm_client(
        _config(provider="gemini", api_key="sk-test", model="gemini-3.1-flash-lite"),
        alternate_keys={"openai": "", "gemini": "sk-test"},
    )

    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-3.1-flash-lite"


def test_resolve_llm_client_prefers_explicit_gemini_provider(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("LABPILOT_LLM_PROVIDER", "gemini")

    client = resolve_llm_client(
        _config(provider="openai", api_key="sk-test"),
        alternate_keys={"openai": "sk-test", "gemini": "gemini-key"},
    )

    assert isinstance(client, GeminiClient)


def test_complete_with_fallback_retries_same_client_before_falling_back(monkeypatch):
    # Free-tier providers (Gemini in particular) frequently return transient
    # 503/429 errors that clear up within seconds; `max_attempts > 1` should
    # retry the *same* client rather than immediately moving on.
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        "labpilot.llm.ollama.OllamaProvider.is_reachable",
        lambda self, timeout_seconds=0.25: False,
    )

    class FlakyThenWorking:
        model = "gemini-3.1-flash-lite"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return "real llm text"

    client = FlakyThenWorking()
    result = complete_with_fallback(
        _config(provider="gemini", api_key="fake-key"),
        "system",
        "user",
        client,
        max_attempts=3,
    )

    assert result == "real llm text"
    assert client.calls == 3


def test_complete_with_fallback_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(
        "labpilot.llm.ollama.OllamaProvider.is_reachable",
        lambda self, timeout_seconds=0.25: False,
    )

    class AlwaysFails:
        model = "gemini-3.1-flash-lite"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, system: str, user: str) -> str:
            self.calls += 1
            raise RuntimeError("still unavailable")

    client = AlwaysFails()
    result = complete_with_fallback(
        _config(provider="gemini", api_key="fake-key"),
        "system",
        "user",
        client,
        max_attempts=3,
    )

    assert result is None
    assert client.calls == 3


def test_complete_with_fallback_tries_alternate_provider_on_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setattr(
        "labpilot.llm.ollama.OllamaProvider.is_reachable",
        lambda self, timeout_seconds=0.25: False,
    )

    class FailingOpenAI:
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str) -> str:
            raise RuntimeError("quota exceeded")

    class WorkingGemini:
        model = "gemini-3.5-flash-lite"

        def complete(self, system: str, user: str) -> str:
            return "gemini narrative"

    config = _config(provider="openai", api_key="sk-test")

    def fake_create(cfg: LLMConfig):
        if cfg.provider == "gemini":
            return WorkingGemini()
        return None

    import labpilot.llm.client as llm_module

    original_create = llm_module.create_llm_client
    llm_module.create_llm_client = fake_create
    try:
        result = complete_with_fallback(
            config,
            "system",
            "user",
            FailingOpenAI(),
        )
    finally:
        llm_module.create_llm_client = original_create

    assert result == "gemini narrative"
