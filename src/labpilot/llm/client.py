import logging
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
    """Try the configured provider, then fall back to the other if it has a key."""
    client = create_llm_client(config)
    if client is not None:
        return client

    if alternate_keys is None:
        from labpilot.config import Settings

        settings = Settings()
        alternate_keys = {
            "openai": settings.openai_api_key,
            "gemini": settings.gemini_api_key,
        }

    primary = config.provider.strip().lower()
    for provider in _CLIENT_BY_PROVIDER:
        if provider == primary:
            continue
        alt_key = (alternate_keys.get(provider) or "").strip()
        if not alt_key:
            continue
        alt_config = config.model_copy(
            update={
                "provider": provider,
                "model": DEFAULT_MODEL_BY_PROVIDER.get(provider, config.model),
                "api_key": alt_key,
            }
        )
        client = create_llm_client(alt_config)
        if client is not None:
            logger.info(
                "LLM provider '%s' unavailable; using '%s' instead.",
                primary,
                provider,
            )
            return client
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
