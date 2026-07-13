import logging
import time
from typing import Protocol

from labpilot.config import DEFAULT_MODEL_BY_PROVIDER, LLMConfig

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal system+user prompt -> text completion interface.

    Keeping this to one method means `BriefGenerator`/`ReflectionGenerator`
    never need to know which provider is actually active.
    """

    def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        from openai import OpenAI  # optional dependency, see pyproject `llm` extra

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float) -> None:
        from google import genai  # optional dependency, see pyproject `llm` extra
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._generate_content_config = types.GenerateContentConfig
        self.model = model
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._generate_content_config(
                system_instruction=system,
                temperature=self.temperature,
            ),
        )
        return response.text or ""


_CLIENT_BY_PROVIDER: dict[str, type] = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def _provider_priority(config: LLMConfig, settings) -> list[str]:
    """Order providers to try: explicit env override, then config, then alternate."""
    explicit = (settings.labpilot_llm_provider or "").strip().lower()
    primary = explicit or config.provider.strip().lower()
    order = [primary] if primary in _CLIENT_BY_PROVIDER else []
    for provider in _CLIENT_BY_PROVIDER:
        if provider not in order:
            order.append(provider)
    return order


def _client_provider_name(client: LLMClient) -> str:
    if isinstance(client, OpenAIClient):
        return "openai"
    if isinstance(client, GeminiClient):
        return "gemini"
    return type(client).__name__


def _llm_config_for_provider(
    config: LLMConfig,
    provider: str,
    alternate_keys: dict[str, str],
) -> LLMConfig:
    # Only substitute the per-provider default model when actually falling
    # back to a *different* provider than the one configured — `config.model`
    # there was chosen for the original provider (e.g. an OpenAI model name)
    # and would be nonsensical to send to a different API. When `provider`
    # *is* the configured provider, the user's explicit `config.model` (e.g.
    # `LABPILOT_LLM_MODEL`) must be respected as-is, not silently replaced
    # with a hardcoded default that may have very different quota/pricing.
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
    """Build the LLM client for `config.provider`, or `None` if it can't be
    used right now.

    Callers must treat `None` as "fall back to template text", never as an
    error — an LLM is an enhancement in P0, not a requirement, so neither a
    missing API key nor a missing optional package (`openai`/`google-genai`)
    should ever raise here.
    """
    provider = config.provider.strip().lower()
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

    model = config.model or DEFAULT_MODEL_BY_PROVIDER.get(provider, "")
    try:
        return client_cls(config.api_key, model, config.temperature)
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
    if alternate_keys is None:
        from labpilot.config import Settings

        settings = Settings()
        alternate_keys = {
            "openai": settings.openai_api_key,
            "gemini": settings.gemini_api_key,
        }
    else:
        from labpilot.config import Settings

        settings = Settings()

    for provider in _provider_priority(config, settings):
        provider_config = _llm_config_for_provider(config, provider, alternate_keys)
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
    """Call LLM providers in priority order until one succeeds.

    Free-tier providers (Gemini in particular) frequently return transient
    `503 UNAVAILABLE` ("high demand") or `429 RESOURCE_EXHAUSTED` (per-minute
    rate limit) errors that clear up within seconds — a single failure there
    is not a real signal that the provider is unusable. When callers want to
    avoid silently downgrading to fallback template text on the first blip,
    they can pass `max_attempts > 1` to retry the *same* provider with a
    fixed backoff before moving on to the next provider (or giving up).
    Defaults to no retry (`max_attempts=1`), preserving prior behavior.
    """
    from labpilot.config import Settings

    settings = Settings()
    alternate_keys = {
        "openai": settings.openai_api_key,
        "gemini": settings.gemini_api_key,
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
                    "Calling LLM provider: %s (attempt %d/%d)", provider_name, attempt, max_attempts
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
    from labpilot.config import Settings

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

    if provider == "openai":
        if not has_openai and has_gemini:
            hints.append(
                "Default provider is OpenAI but only GEMINI_API_KEY is set — "
                "add `LABPILOT_LLM_PROVIDER=gemini` to .env"
            )
        elif not has_openai:
            hints.append("Set OPENAI_API_KEY in .env, or switch to Gemini")
    elif provider == "gemini":
        if not has_gemini:
            hints.append("Set GEMINI_API_KEY in .env")

    if has_openai and has_gemini and provider == "openai":
        hints.append(
            "Both keys are set; to prefer Gemini add `LABPILOT_LLM_PROVIDER=gemini` to .env"
        )

    hints.append("Or pass `--yes` to continue with template-only brief/reflection")
    return hints
