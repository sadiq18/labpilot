"""Task → provider/model routing.

Extension points for Claude / OpenRouter: add providers when clients land —
callers never import them directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from labpilot.config import DEFAULT_MODEL_BY_PROVIDER, LLMConfig, Settings, TaskProfile
from labpilot.llm.budget import BudgetLedger
from labpilot.llm.catalog import ProviderSpec, RoutingConfig, eligible_providers
from labpilot.llm.schemas import ResolvedRoute

logger = logging.getLogger(__name__)

CLOUD_PROVIDERS = frozenset({"openai", "gemini"})
# Future: anthropic, openrouter
LOCAL_PROVIDER = "ollama"


def _task_profile(config: LLMConfig, task: str) -> TaskProfile:
    if task in config.tasks:
        return config.tasks[task]
    if "default" in config.tasks:
        return config.tasks["default"]
    return TaskProfile()


def _has_cloud_key(provider: str, config: LLMConfig, settings: Settings) -> bool:
    if provider == "openai":
        key = settings.openai_api_key.strip()
        if not key and config.provider.strip().lower() == "openai":
            key = config.api_key.strip()
        return bool(key)
    if provider == "gemini":
        key = settings.gemini_api_key.strip()
        if not key and config.provider.strip().lower() == "gemini":
            key = config.api_key.strip()
        return bool(key)
    return False


def _package_available(provider: str) -> bool:
    try:
        if provider == "openai":
            import openai  # noqa: F401
        elif provider == "gemini":
            import google.genai  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


def cloud_available(config: LLMConfig, settings: Settings | None = None) -> str | None:
    """Return first usable cloud provider name, preferring config.provider."""
    settings = settings or Settings()
    preferred = config.provider.strip().lower()
    order: list[str] = []
    if preferred in CLOUD_PROVIDERS:
        order.append(preferred)
    for name in CLOUD_PROVIDERS:
        if name not in order:
            order.append(name)
    for name in order:
        if _has_cloud_key(name, config, settings) and _package_available(name):
            return name
    return None


def ollama_reachable(config: LLMConfig) -> bool:
    from labpilot.llm.ollama import OllamaProvider

    return OllamaProvider(config.ollama_base_url).is_reachable()


def _ollama_model(config: LLMConfig, profile: TaskProfile) -> str:
    if profile.model:
        return profile.model
    if config.provider.strip().lower() == LOCAL_PROVIDER and config.model:
        return config.model
    return config.fallback_model or DEFAULT_MODEL_BY_PROVIDER[LOCAL_PROVIDER]


def _cloud_model(config: LLMConfig, profile: TaskProfile, provider: str) -> str:
    if profile.model:
        return profile.model
    if provider == config.provider.strip().lower() and config.model:
        return config.model
    return DEFAULT_MODEL_BY_PROVIDER.get(provider, config.model)


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of tiered routing.

    ``provider`` is None when nothing is currently usable; ``wait_seconds``
    then says how long until the best candidate frees up. Callers pace on that
    rather than firing a call they know will be rejected.
    """

    provider: ProviderSpec | None
    model: str = ""
    role: str = ""
    wait_seconds: float = 0.0
    degraded: bool = False
    reason: str = ""


def select_route(
    routing: RoutingConfig,
    role: str,
    ledger: BudgetLedger,
    *,
    now: float | None = None,
) -> RouteDecision:
    """Pick a provider for ``role`` within entitlement, policy and budget.

    Exhaustion is handled per role rather than globally. Downgrading a codegen
    or hypothesis call to a weak model is worse than waiting: the weak output
    gets recorded as "this technique did not help", a false negative that is
    indistinguishable from a real one and permanently pollutes research memory.
    Summarisation has no such property and degrades freely.
    """
    spec = routing.role_spec(role)
    candidates = eligible_providers(routing, role)

    best_wait: float | None = None
    best_reason = ""
    for provider in candidates:
        avail = ledger.availability(
            provider.name,
            rpm=provider.rpm,
            rpd=provider.rpd,
            tpm=provider.tpm,
            now=now,
        )
        if avail.ok:
            return RouteDecision(
                provider=provider,
                model=provider.model_for(role) or "",
                role=role,
                reason=f"{provider.tier} tier",
            )
        if best_wait is None or avail.wait_seconds < best_wait:
            best_wait, best_reason = avail.wait_seconds, avail.reason

    if not candidates:
        best_reason = "no eligible provider (entitlement, data policy, or credentials)"

    if spec.on_exhaustion == "degrade":
        for provider in eligible_providers(routing, role, ignore_strength=True):
            avail = ledger.availability(
                provider.name,
                rpm=provider.rpm,
                rpd=provider.rpd,
                tpm=provider.tpm,
                now=now,
            )
            if avail.ok:
                return RouteDecision(
                    provider=provider,
                    model=provider.model_for(role) or "",
                    role=role,
                    degraded=True,
                    reason=f"degraded to {provider.tier} tier: {best_reason}",
                )

    return RouteDecision(
        provider=None,
        role=role,
        wait_seconds=best_wait or 0.0,
        reason=best_reason,
    )


def resolve_route(
    config: LLMConfig,
    task: str,
    *,
    temperature: float | None = None,
    settings: Settings | None = None,
    ollama_ok: bool | None = None,
) -> ResolvedRoute | None:
    """Pick provider+model for ``task``.

    1. ``force_local`` → always Ollama
    2. ``mode: local`` → Ollama
    3. ``mode: cloud`` → cloud only
    4. ``mode: auto`` → cloud if available else Ollama if reachable
    5. Missing task → ``default`` profile / global config
    """
    settings = settings or Settings()
    profile = _task_profile(config, task)
    mode = (config.mode or "auto").strip().lower()
    temp = (
        temperature
        if temperature is not None
        else profile.temperature
        if profile.temperature is not None
        else config.temperature
    )
    profile_provider = (profile.provider or "").strip().lower() or None

    def make(provider: str, model: str) -> ResolvedRoute:
        return ResolvedRoute(provider=provider, model=model, temperature=temp, task=task)

    def try_ollama() -> ResolvedRoute | None:
        reachable = ollama_ok if ollama_ok is not None else ollama_reachable(config)
        if not reachable:
            return None
        return make(LOCAL_PROVIDER, _ollama_model(config, profile))

    def try_cloud(preferred: str | None = None) -> ResolvedRoute | None:
        if preferred in CLOUD_PROVIDERS:
            if _has_cloud_key(preferred, config, settings) and _package_available(preferred):
                return make(preferred, _cloud_model(config, profile, preferred))
        cloud = cloud_available(config, settings)
        if cloud is None:
            return None
        return make(cloud, _cloud_model(config, profile, cloud))

    # 1) Force local always wins.
    if profile.force_local:
        return make(LOCAL_PROVIDER, _ollama_model(config, profile))

    # 2) Global local mode.
    if mode == "local":
        return make(LOCAL_PROVIDER, _ollama_model(config, profile))

    # 3) Cloud-only mode.
    if mode == "cloud":
        route = try_cloud(profile_provider if profile_provider in CLOUD_PROVIDERS else None)
        if route is None:
            logger.info("LLM mode=cloud but no cloud provider available for task=%s", task)
        return route

    # 4) auto
    if profile_provider == LOCAL_PROVIDER:
        return try_ollama() or try_cloud()

    if profile_provider in CLOUD_PROVIDERS:
        return try_cloud(profile_provider) or try_ollama()

    if config.provider.strip().lower() == LOCAL_PROVIDER:
        return try_ollama() or try_cloud()

    return try_cloud() or try_ollama()
