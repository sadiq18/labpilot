"""Public LLM entry point + backward-compatible client helpers.

Callers should prefer::

    from labpilot.llm import LLM
    llm = LLM(config)
    result = llm.generate(task="planning", prompt=..., response_model=...)

Legacy ``LLMClient.complete(system, user)`` remains for Micro Agents.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol, TypeVar, overload

from pydantic import BaseModel, ValidationError

from labpilot.config import DEFAULT_MODEL_BY_PROVIDER, LLMConfig, Settings
from labpilot.llm.cache import PromptCache, cache_key
from labpilot.llm.json_utils import parse_json_object
from labpilot.llm.ollama import OllamaClient, OllamaProvider
from labpilot.llm.providers import GeminiClient, GeminiProvider, OpenAIClient, OpenAIProvider
from labpilot.llm.router import LOCAL_PROVIDER, cloud_available, resolve_route
from labpilot.llm.schemas import ResolvedRoute

logger = logging.getLogger(__name__)

# Re-export for tests / callers that historically imported from this module.
__all__ = [
    "DEFAULT_MODEL_BY_PROVIDER",
    "LLM",
    "LLMClient",
    "OpenAIClient",
    "GeminiClient",
    "OllamaClient",
    "complete_with_fallback",
    "create_llm_client",
    "llm_setup_hints",
    "resolve_llm_client",
]

TModel = TypeVar("TModel", bound=BaseModel)

_PARSE_RETRIES = 2
_LOWERED_TEMPERATURE = 0.0


class LLMClient(Protocol):
    """Minimal system+user prompt -> text completion interface."""

    def complete(self, system: str, user: str) -> str: ...


_CLIENT_BY_PROVIDER: dict[str, type] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
}


def _provider_priority(config: LLMConfig, settings: Settings) -> list[str]:
    """Order providers to try for legacy resolve/complete_with_fallback."""
    mode = (config.mode or "auto").strip().lower()
    explicit = (settings.labpilot_llm_provider or "").strip().lower()
    primary = explicit or config.provider.strip().lower()

    if mode == "local" or primary == LOCAL_PROVIDER:
        order = [LOCAL_PROVIDER]
        for provider in ("openai", "gemini"):
            if provider not in order:
                order.append(provider)
        return order

    order = [primary] if primary in _CLIENT_BY_PROVIDER else []
    # Prefer cloud then ollama for auto/cloud.
    for provider in ("openai", "gemini", LOCAL_PROVIDER):
        if provider not in order:
            if mode == "cloud" and provider == LOCAL_PROVIDER:
                continue
            order.append(provider)
    return order


def _client_provider_name(client: LLMClient) -> str:
    if isinstance(client, OpenAIClient):
        return "openai"
    if isinstance(client, GeminiClient):
        return "gemini"
    if isinstance(client, OllamaClient):
        return LOCAL_PROVIDER
    return type(client).__name__


def _llm_config_for_provider(
    config: LLMConfig,
    provider: str,
    alternate_keys: dict[str, str],
) -> LLMConfig:
    is_primary_provider = provider == config.provider.strip().lower()
    model = (
        config.model
        if is_primary_provider and config.model
        else DEFAULT_MODEL_BY_PROVIDER.get(provider, config.model)
    )
    return config.model_copy(
        update={
            "provider": provider,
            "model": model,
            "api_key": (alternate_keys.get(provider) or "").strip(),
        }
    )


def create_llm_client(config: LLMConfig) -> LLMClient | None:
    """Build the LLM client for ``config.provider``, or ``None`` if unusable.

    Callers must treat ``None`` as soft-fail (templates / rule_engine).
    """
    provider = config.provider.strip().lower()
    model = config.model or DEFAULT_MODEL_BY_PROVIDER.get(provider, "")

    if provider == LOCAL_PROVIDER:
        try:
            return OllamaClient(
                config.ollama_base_url,
                model,
                config.temperature,
                config.request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to construct Ollama client (%s); soft-failing.", exc)
            return None

    client_cls = _CLIENT_BY_PROVIDER.get(provider)
    if client_cls is None:
        logger.warning(
            "Unknown LLM provider '%s' (expected one of: %s); using fallback template text.",
            config.provider,
            ", ".join(sorted(_CLIENT_BY_PROVIDER)),
        )
        return None

    if not config.api_key:
        logger.info(
            "No API key configured for LLM provider '%s'; using fallback template text.",
            provider,
        )
        return None

    try:
        return client_cls(config.api_key, model, config.temperature)  # type: ignore[call-arg]
    except ImportError as exc:
        logger.warning(
            "LLM provider '%s' needs an optional package that isn't installed (%s); "
            "install it with `uv sync --extra llm`, or continue without an LLM key. "
            "Using fallback template text.",
            provider,
            exc,
        )
        return None


def resolve_llm_client(
    config: LLMConfig,
    *,
    alternate_keys: dict[str, str] | None = None,
) -> LLMClient | None:
    """Return the first LLM client that can be constructed for a configured provider."""
    settings = Settings()
    if alternate_keys is None:
        alternate_keys = {
            "openai": settings.openai_api_key,
            "gemini": settings.gemini_api_key,
            "ollama": "",
        }

    mode = (config.mode or "auto").strip().lower()
    if mode == "auto" and not cloud_available(config, settings):
        # Prefer Ollama when no cloud keys.
        ollama_config = _llm_config_for_provider(config, LOCAL_PROVIDER, alternate_keys)
        if OllamaProvider(config.ollama_base_url).is_reachable():
            client = create_llm_client(ollama_config)
            if client is not None:
                logger.info("Using LLM provider: ollama")
                return client

    for provider in _provider_priority(config, settings):
        provider_config = _llm_config_for_provider(config, provider, alternate_keys)
        if provider == LOCAL_PROVIDER:
            if mode == "cloud":
                continue
            if not OllamaProvider(config.ollama_base_url).is_reachable():
                continue
        client = create_llm_client(provider_config)
        if client is not None:
            if provider != config.provider.strip().lower():
                logger.info(
                    "LLM provider '%s' unavailable; using '%s' instead.",
                    config.provider,
                    provider,
                )
            else:
                logger.info("Using LLM provider: %s", provider)
            return client
    return None


def complete_with_fallback(
    config: LLMConfig,
    system: str,
    user: str,
    llm_client: LLMClient | None = None,
    *,
    max_attempts: int = 1,
    retry_delay_seconds: float = 20.0,
) -> str | None:
    """Call LLM providers in priority order until one succeeds."""
    settings = Settings()
    alternate_keys = {
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
        "ollama": "",
    }

    clients: list[LLMClient] = []
    seen_providers: set[str] = set()

    if llm_client is not None:
        clients.append(llm_client)
        seen_providers.add(_client_provider_name(llm_client))

    for provider in _provider_priority(config, settings):
        if provider in seen_providers:
            continue
        provider_config = _llm_config_for_provider(config, provider, alternate_keys)
        if provider == LOCAL_PROVIDER and not OllamaProvider(config.ollama_base_url).is_reachable():
            continue
        client = create_llm_client(provider_config)
        if client is None:
            continue
        clients.append(client)
        seen_providers.add(provider)

    for client in clients:
        provider_name = _client_provider_name(client)
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "Calling LLM provider: %s (attempt %d/%d)",
                    provider_name,
                    attempt,
                    max_attempts,
                )
                return client.complete(system, user)
            except Exception as exc:
                if attempt < max_attempts:
                    logger.warning(
                        "LLM call failed for provider '%s' (%s); retrying in %.0fs "
                        "(attempt %d/%d).",
                        provider_name,
                        exc,
                        retry_delay_seconds,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(retry_delay_seconds)
                else:
                    logger.warning(
                        "LLM call failed for provider '%s' (%s); trying next provider if "
                        "available.",
                        provider_name,
                        exc,
                    )
    return None


def llm_setup_hints(config: LLMConfig) -> list[str]:
    """Actionable hints when no LLM client could be created."""
    settings = Settings()
    provider = config.provider.strip().lower()
    hints: list[str] = []

    try:
        import openai  # noqa: F401
    except ImportError:
        openai_installed = False
    else:
        openai_installed = True

    try:
        import google.genai  # noqa: F401
    except ImportError:
        gemini_installed = False
    else:
        gemini_installed = True

    if not openai_installed or not gemini_installed:
        hints.append("Install LLM packages: `uv sync --extra llm`")

    has_openai = bool(settings.openai_api_key.strip())
    has_gemini = bool(settings.gemini_api_key.strip())
    ollama_up = OllamaProvider(config.ollama_base_url).is_reachable()

    if provider == "openai":
        if not has_openai and has_gemini:
            hints.append(
                "Default provider is OpenAI but only GEMINI_API_KEY is set — "
                "add `LABPILOT_LLM_PROVIDER=gemini` to .env"
            )
        elif not has_openai and not ollama_up:
            hints.append(
                "Set OPENAI_API_KEY in .env, switch to Gemini, or run Ollama locally "
                f"({config.ollama_base_url}) with `ollama pull {config.fallback_model}`"
            )
        elif not has_openai:
            hints.append("Set OPENAI_API_KEY in .env, or switch to Gemini / Ollama")
    elif provider == "gemini":
        if not has_gemini:
            hints.append("Set GEMINI_API_KEY in .env")
    elif provider == LOCAL_PROVIDER and not ollama_up:
        hints.append(
            f"Start Ollama at {config.ollama_base_url} and pull a model "
            f"(e.g. `ollama pull {config.fallback_model}`)"
        )

    if has_openai and has_gemini and provider == "openai":
        hints.append(
            "Both keys are set; to prefer Gemini add `LABPILOT_LLM_PROVIDER=gemini` to .env"
        )

    if not ollama_up and config.mode in {"auto", "local"}:
        hints.append(
            "For free local models: install Ollama, then "
            f"`ollama pull {config.fallback_model}` (see llm.tasks in default.yaml)"
        )

    hints.append("Or pass `--yes` to continue with template-only brief/reflection")
    return hints


def load_prompt(name: str) -> str:
    """Load a markdown system prompt from ``labpilot/llm/prompts/``."""
    path = Path(__file__).resolve().parent / "prompts" / name
    if not path.suffix:
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8").strip()


class _BoundClient:
    """LLMClient adapter that delegates to :class:`LLM.generate` (task=default)."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        route = resolve_route(llm.config, "default", ollama_ok=True)
        self.model = route.model if route else llm.config.model

    def complete(self, system: str, user: str) -> str:
        result = self._llm.generate(task="default", prompt=user, system=system)
        if result is None:
            raise RuntimeError("LLM.generate soft-failed")
        return str(result)


class LLM:
    """Task-routed LLM facade with cache, parse retry, and Ollama fallback."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        cache: PromptCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or Settings()
        if cache is not None:
            self._cache = cache
        else:
            self._cache = PromptCache(
                config.cache.path,
                enabled=config.cache.enabled,
            )

    @overload
    def generate(
        self,
        task: str,
        prompt: str,
        *,
        system: str = "",
        response_model: None = None,
        temperature: float | None = None,
    ) -> str | None: ...

    @overload
    def generate(
        self,
        task: str,
        prompt: str,
        *,
        system: str = "",
        response_model: type[TModel],
        temperature: float | None = None,
    ) -> TModel | None: ...

    def generate(
        self,
        task: str,
        prompt: str,
        *,
        system: str = "",
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
    ) -> str | BaseModel | None:
        route = resolve_route(
            self.config,
            task,
            temperature=temperature,
            settings=self.settings,
        )
        if route is None:
            return None

        key = cache_key(route.model, prompt, route.temperature, system)
        cached = self._cache.get(key)
        if cached is not None:
            return self._coerce(cached, response_model)

        text = self._generate_with_retries(
            route,
            system=system,
            prompt=prompt,
            response_model=response_model,
        )
        if text is None:
            return None
        self._cache.set(key, text, model=route.model)
        return self._coerce(text, response_model)

    def as_client(self) -> LLMClient:
        """Legacy ``complete(system, user)`` adapter."""
        return _BoundClient(self)

    def _coerce(
        self,
        text: str,
        response_model: type[BaseModel] | None,
    ) -> str | BaseModel | None:
        if response_model is None:
            return text
        try:
            return response_model.model_validate(parse_json_object(text))
        except (ValueError, ValidationError) as exc:
            logger.warning("Cached/parsed LLM response failed validation (%s)", exc)
            return None

    def _generate_with_retries(
        self,
        route: ResolvedRoute,
        *,
        system: str,
        prompt: str,
        response_model: type[BaseModel] | None,
    ) -> str | None:
        attempts: list[tuple[str, str, float]] = [
            (route.provider, route.model, route.temperature),
        ]
        # Retry same model at lowered temperature after parse failures.
        if route.temperature > _LOWERED_TEMPERATURE:
            attempts.append((route.provider, route.model, _LOWERED_TEMPERATURE))
        # Final fallback: local Ollama fallback_model.
        fallback_model = self.config.fallback_model or DEFAULT_MODEL_BY_PROVIDER[LOCAL_PROVIDER]
        if not (route.provider == LOCAL_PROVIDER and route.model == fallback_model):
            attempts.append((LOCAL_PROVIDER, fallback_model, _LOWERED_TEMPERATURE))

        last_raw: str | None = None
        for provider_name, model, temp in attempts:
            provider = self._build_provider(provider_name)
            if provider is None:
                continue
            for parse_attempt in range(1, _PARSE_RETRIES + 1):
                try:
                    logger.info(
                        "LLM generate task=%s provider=%s model=%s temp=%.2f attempt=%d",
                        route.task,
                        provider_name,
                        model,
                        temp,
                        parse_attempt,
                    )
                    raw = provider.complete(system, prompt, model=model, temperature=temp)
                    last_raw = raw
                    if response_model is None:
                        return raw
                    # Validate parse; retry on failure.
                    response_model.model_validate(parse_json_object(raw))
                    return raw
                except Exception as exc:  # noqa: BLE001 — never leak raw failures
                    logger.warning(
                        "LLM generate failed (provider=%s model=%s): %s",
                        provider_name,
                        model,
                        exc,
                    )
                    continue
        if last_raw is not None and response_model is None:
            return last_raw
        return None

    def _build_provider(self, provider_name: str):
        name = provider_name.strip().lower()
        try:
            if name == "openai":
                key = self.settings.openai_api_key.strip() or (
                    self.config.api_key if self.config.provider == "openai" else ""
                )
                if not key:
                    return None
                return OpenAIProvider(key)
            if name == "gemini":
                key = self.settings.gemini_api_key.strip() or (
                    self.config.api_key if self.config.provider == "gemini" else ""
                )
                if not key:
                    return None
                return GeminiProvider(key)
            if name == LOCAL_PROVIDER:
                return OllamaProvider(self.config.ollama_base_url)
        except ImportError as exc:
            logger.warning("Provider '%s' package missing (%s)", name, exc)
            return None
        logger.warning("Unknown provider '%s'", name)
        return None
