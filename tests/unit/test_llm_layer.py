"""Unit tests for task router, prompt cache, Ollama provider, and LLM.generate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from labpilot.config import LLMCacheConfig, LLMConfig, TaskProfile
from fitroute.cache import PromptCache, cache_key
from labpilot.llm.client import LLM
from labpilot.llm.ollama import OllamaProvider
from labpilot.llm.router import resolve_route

_OLLAMA_DEFAULT = "qwen2.5-coder:14b"


class _TinyModel(BaseModel):
    answer: str = Field(default="")


def _config(**kwargs) -> LLMConfig:
    base = dict(
        mode="auto",
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
        api_key="",
        ollama_base_url="http://localhost:11434",
        fallback_model=_OLLAMA_DEFAULT,
        cache=LLMCacheConfig(enabled=False, path=Path(".cache/llm-test.sqlite")),
        tasks={
            "planning": TaskProfile(
                model=_OLLAMA_DEFAULT,
                provider="ollama",
                force_local=True,
            ),
            "coding": TaskProfile(
                model=_OLLAMA_DEFAULT,
                provider="ollama",
                force_local=True,
            ),
            "summary": TaskProfile(model=_OLLAMA_DEFAULT, provider="ollama"),
            "default": TaskProfile(),
        },
    )
    base.update(kwargs)
    return LLMConfig(**base)


# --- router -----------------------------------------------------------------


def test_router_force_local_always_ollama():
    route = resolve_route(_config(mode="cloud"), "planning", ollama_ok=False)
    assert route is not None
    assert route.provider == "ollama"
    assert route.model == _OLLAMA_DEFAULT


def test_router_mode_local_uses_ollama():
    route = resolve_route(_config(mode="local"), "default", ollama_ok=False)
    assert route is not None
    assert route.provider == "ollama"
    assert route.model == _OLLAMA_DEFAULT


def test_router_mode_cloud_returns_none_without_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from labpilot.config import Settings

    settings = Settings(openai_api_key="", gemini_api_key="")
    route = resolve_route(
        _config(mode="cloud", api_key=""),
        "default",
        settings=settings,
        ollama_ok=True,
    )
    assert route is None


def test_router_auto_prefers_cloud_when_key_and_package(monkeypatch):
    pytest.importorskip("openai")
    from labpilot.config import Settings

    settings = Settings(openai_api_key="sk-test", gemini_api_key="")
    route = resolve_route(
        _config(mode="auto", api_key="sk-test"),
        "default",
        settings=settings,
        ollama_ok=True,
    )
    assert route is not None
    assert route.provider == "openai"
    assert route.model == "gpt-4o-mini"


def test_router_auto_falls_back_to_ollama_without_cloud(monkeypatch):
    from labpilot.config import Settings

    settings = Settings(openai_api_key="", gemini_api_key="")
    route = resolve_route(
        _config(mode="auto", api_key=""),
        "default",
        settings=settings,
        ollama_ok=True,
    )
    assert route is not None
    assert route.provider == "ollama"
    assert route.model == _OLLAMA_DEFAULT


def test_router_summary_task_prefers_ollama_when_reachable():
    from labpilot.config import Settings

    settings = Settings(openai_api_key="", gemini_api_key="")
    route = resolve_route(
        _config(mode="auto"),
        "summary",
        settings=settings,
        ollama_ok=True,
    )
    assert route is not None
    assert route.provider == "ollama"
    assert route.model == _OLLAMA_DEFAULT


# --- cache ------------------------------------------------------------------


def test_cache_hit_miss_key_stability(tmp_path: Path):
    db = tmp_path / "llm.sqlite"
    cache = PromptCache(db, enabled=True)
    key = cache_key("m", "prompt", 0.3, "sys")
    assert cache.get(key) is None
    cache.set(key, "hello", model="m")
    assert cache.get(key) == "hello"
    # Same inputs → same key
    assert cache_key("m", "prompt", 0.3, "sys") == key
    assert cache_key("m", "prompt", 0.4, "sys") != key
    cache.close()


# --- ollama provider --------------------------------------------------------


def test_ollama_provider_builds_chat_request(monkeypatch):
    captured: dict = {}

    class _Resp:
        def read(self) -> bytes:
            return json.dumps({"message": {"content": "ok"}}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr("labpilot.llm.ollama.urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider("http://localhost:11434")
    text = provider.complete("sys", "user", model=_OLLAMA_DEFAULT, temperature=0.2)
    assert text == "ok"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == _OLLAMA_DEFAULT
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["body"]["options"]["temperature"] == 0.2


# --- LLM.generate -----------------------------------------------------------


def test_generate_returns_text_and_uses_cache(tmp_path: Path, monkeypatch):
    from labpilot.config import Settings

    calls: list[tuple] = []

    class FakeProvider:
        def complete(self, system, user, *, model, temperature):
            calls.append((system, user, model, temperature))
            return "cached-or-fresh"

    monkeypatch.setattr(
        LLM,
        "_build_provider",
        lambda self, name: FakeProvider(),
    )
    # Avoid real Ollama probe in resolve_route for force_local tasks.
    cfg = _config(
        cache=LLMCacheConfig(enabled=True, path=tmp_path / "c.sqlite"),
    )
    llm = LLM(cfg, settings=Settings(openai_api_key="", gemini_api_key=""))
    first = llm.generate(task="planning", prompt="p", system="s")
    second = llm.generate(task="planning", prompt="p", system="s")
    assert first == "cached-or-fresh"
    assert second == "cached-or-fresh"
    assert len(calls) == 1  # second served from cache


def test_generate_response_model_parse_success(tmp_path: Path, monkeypatch):
    from labpilot.config import Settings

    class FakeProvider:
        def complete(self, system, user, *, model, temperature):
            return '{"answer": "yes"}'

    monkeypatch.setattr(LLM, "_build_provider", lambda self, name: FakeProvider())
    llm = LLM(
        _config(cache=LLMCacheConfig(enabled=False, path=tmp_path / "c.sqlite")),
        settings=Settings(),
    )
    result = llm.generate(
        task="planning",
        prompt="p",
        system="s",
        response_model=_TinyModel,
    )
    assert isinstance(result, _TinyModel)
    assert result.answer == "yes"


def test_generate_retries_then_fallback_model(tmp_path: Path, monkeypatch):
    from labpilot.config import Settings

    calls: list[tuple[str, str, float]] = []
    primary = "codellama:7b"

    class FlakyThenFallback:
        def complete(self, system, user, *, model, temperature):
            calls.append((model, "x", temperature))
            if model != _OLLAMA_DEFAULT:
                raise RuntimeError("primary down")
            return '{"answer": "fallback"}'

    # Route coding to a non-fallback ollama model so fallback path is distinct.
    cfg = _config(
        tasks={
            "coding": TaskProfile(
                model=primary,
                provider="ollama",
                force_local=True,
                temperature=0.5,
            )
        },
        cache=LLMCacheConfig(enabled=False, path=tmp_path / "c.sqlite"),
        fallback_model=_OLLAMA_DEFAULT,
    )
    monkeypatch.setattr(LLM, "_build_provider", lambda self, name: FlakyThenFallback())
    llm = LLM(cfg, settings=Settings())
    result = llm.generate(
        task="coding",
        prompt="p",
        system="s",
        response_model=_TinyModel,
    )
    assert isinstance(result, _TinyModel)
    assert result.answer == "fallback"
    # Primary model attempted (possibly with lowered temp), then fallback model.
    models_used = [c[0] for c in calls]
    assert primary in models_used
    assert _OLLAMA_DEFAULT in models_used


def test_generate_parse_failure_returns_none(tmp_path: Path, monkeypatch):
    from labpilot.config import Settings

    class BadProvider:
        def complete(self, system, user, *, model, temperature):
            return "not-json"

    monkeypatch.setattr(LLM, "_build_provider", lambda self, name: BadProvider())
    llm = LLM(
        _config(cache=LLMCacheConfig(enabled=False, path=tmp_path / "c.sqlite")),
        settings=Settings(),
    )
    result = llm.generate(
        task="planning",
        prompt="p",
        response_model=_TinyModel,
    )
    assert result is None


def test_create_llm_client_ollama_without_api_key():
    from labpilot.llm.client import OllamaClient, create_llm_client

    client = create_llm_client(
        _config(provider="ollama", model=_OLLAMA_DEFAULT, api_key="")
    )
    assert isinstance(client, OllamaClient)
    assert client.model == _OLLAMA_DEFAULT


# --- JSON extraction robustness (local models are messy) --------------------


def test_parse_json_object_plain():
    from labpilot.llm.json_utils import parse_json_object

    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_json_object_markdown_fence_with_prose():
    from labpilot.llm.json_utils import parse_json_object

    text = 'Sure! Here is the result:\n```json\n{"tool": "run_plan"}\n```\nHope that helps.'
    assert parse_json_object(text) == {"tool": "run_plan"}


def test_parse_json_object_survives_prose_containing_braces():
    """Naive first-brace/last-brace slicing spans the prose and parses as neither."""
    from labpilot.llm.json_utils import parse_json_object

    text = 'Use the {placeholder} format. Answer: {"decision": "approve"} done.'
    assert parse_json_object(text) == {"decision": "approve"}


def test_parse_json_object_picks_an_object_when_several_are_emitted():
    from labpilot.llm.json_utils import parse_json_object

    text = '{"first": 1}\n{"second": 2, "extra": "longer object wins"}'
    assert parse_json_object(text)["second"] == 2


def test_parse_json_object_tolerates_trailing_comma():
    from labpilot.llm.json_utils import parse_json_object

    assert parse_json_object('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_parse_json_object_keeps_nested_braces_in_strings():
    from labpilot.llm.json_utils import parse_json_object

    assert parse_json_object('{"note": "a } brace in a string"}') == {
        "note": "a } brace in a string"
    }


def test_parse_json_object_error_includes_the_response():
    import pytest

    from labpilot.llm.json_utils import parse_json_object

    with pytest.raises(ValueError, match="I cannot help"):
        parse_json_object("I cannot help with that request.")


# --- constrained JSON decoding ----------------------------------------------


def test_ollama_requests_json_format_when_asked(monkeypatch):
    """Small local models ignore 'reply with JSON'; the runtime must enforce it."""
    import json as _json

    from labpilot.llm.ollama import OllamaProvider

    captured = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps({"message": {"content": "{}"}}).encode()

    def _fake_urlopen(request, timeout=None):
        captured["payload"] = _json.loads(request.data.decode())
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    provider = OllamaProvider("http://127.0.0.1:11434")

    provider.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert captured["payload"]["format"] == "json"

    provider.complete("s", "u", model="m", temperature=0.1)
    assert "format" not in captured["payload"]


def test_micro_agent_helper_falls_back_for_clients_without_json_mode():
    from labpilot.accessor.common.micro_agents import _complete_json

    class _Old:
        def complete(self, system, user):
            return '{"ok": true}'

    class _New:
        def complete(self, system, user, *, json_mode=False):
            return '{"json_mode": %s}' % ("true" if json_mode else "false")

    assert _complete_json(_Old(), "s", "u") == '{"ok": true}'
    assert _complete_json(_New(), "s", "u") == '{"json_mode": true}'
