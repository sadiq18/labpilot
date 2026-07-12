import sys

import pytest

from labpilot.config import LLMConfig
from labpilot.llm import client as llm_client_module
from labpilot.llm.client import GeminiClient, OpenAIClient, create_llm_client, resolve_llm_client


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
    assert client.model == "gemini-3.5-flash"


def test_create_llm_client_respects_explicit_model_override():
    pytest.importorskip("openai")

    client = create_llm_client(_config(provider="openai", api_key="sk-test", model="gpt-4.1"))

    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-4.1"


def test_default_model_by_provider_matches_client_registry():
    assert set(llm_client_module.DEFAULT_MODEL_BY_PROVIDER) == {"openai", "gemini"}


def test_resolve_llm_client_falls_back_to_gemini_when_openai_unavailable(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setitem(sys.modules, "openai", None)

    client = resolve_llm_client(
        _config(provider="openai", api_key="sk-test"),
        alternate_keys={"openai": "sk-test", "gemini": "gemini-key"},
    )

    assert isinstance(client, GeminiClient)
