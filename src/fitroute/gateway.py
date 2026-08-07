"""Execution: cache, call, meter, stamp.

`select()` decides; this executes. The split matters because caching and
metering are cross-cutting, and leaving them to callers is exactly what
produced labpilot's measured state before this existed — one prompt-cache row
across nine campaigns, and no caller of ``BudgetLedger.record`` at all.

If a call goes through :class:`RoleBoundClient`, it is cached, metered and
stamped. There is no second path.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

from fitroute.adapters import build_adapter
from fitroute.budget import BudgetLedger
from fitroute.cache import PromptCache, cache_key
from fitroute.catalog import CredentialResolver, RoutingConfig
from fitroute.select import select_route

logger = logging.getLogger("fitroute.gateway")


class RoleUnavailable(RuntimeError):
    """No provider can serve this role, and waiting would exceed the bound."""


@dataclass(frozen=True)
class ServedBy:
    """What actually served a call. Attribution, not decoration.

    Without this a failed hypothesis cannot be attributed to the idea rather
    than the writer — the distinction this whole layer exists to protect.
    """

    provider: str
    model: str
    tier: str
    role: str
    degraded: bool = False
    cache_hit: bool = False
    tokens: int | None = None
    latency_ms: int = 0

    def __str__(self) -> str:
        bits = [f"{self.provider}/{self.model}", self.tier]
        if self.degraded:
            bits.append("degraded")
        if self.cache_hit:
            bits.append("cached")
        return " · ".join(bits)


class LLMGateway:
    """Hands out role-bound clients. One per process is plenty."""

    def __init__(
        self,
        routing: RoutingConfig,
        ledger: BudgetLedger,
        *,
        cache: PromptCache | None = None,
        credential_resolver: CredentialResolver | None = None,
        temperature: float = 0.3,
    ) -> None:
        self.routing = routing
        self.ledger = ledger
        self.cache = cache
        self.credential_resolver = credential_resolver
        self.temperature = temperature

    def for_role(self, role: str) -> RoleBoundClient:
        return RoleBoundClient(self, role)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Satisfy the plain client contract, using the ``default`` role.

        Lets the gateway be handed to anything that expects a client, so
        adoption is incremental: a call site that has not yet declared a role
        still routes, meters and caches — it just does not get role-specific
        capability requirements.
        """
        return self.for_role("default").complete(system, user, json_mode=json_mode)

    def preview(self, role: str):
        """Resolve without calling — for `doctor` and dry runs."""
        return select_route(
            self.routing,
            role,
            self.ledger,
            credential_resolver=self.credential_resolver,
        )

    def _api_key(self, env_name: str) -> str:
        if not env_name:
            return ""
        if self.credential_resolver is not None:
            resolved = self.credential_resolver(env_name).strip()
            if resolved:
                return resolved
        return os.environ.get(env_name, "").strip()


class _ProviderCallFailed(Exception):
    """Internal: carries which provider failed, so it can be cooled down.

    Never escapes `complete` — the original exception is re-raised so callers
    and `_is_transient_llm_error` see exactly what they saw before.
    """

    def __init__(self, provider: str, original: Exception) -> None:
        super().__init__(str(original))
        self.provider = provider
        self.original = original


#: Upstream conditions worth trying a different provider for. A malformed
#: request or a bad key will fail identically everywhere, so failing over on
#: those just burns the whole chain to reach the same error more slowly.
_RETRYABLE_MARKERS = (
    "429",
    "RATE LIMIT",
    "RESOURCE_EXHAUSTED",
    "QUOTA",
    "HIGH DEMAND",
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "OVERLOADED",
    "TIMEOUT",
    "TIMED OUT",
    "CONNECTION",
    "TEMPORARILY",
    # Measured on rogii 2026-08-07: OpenRouter returns HTTP 200 with an empty
    # `choices` array for some free models. Not an error status, not a rate
    # limit, and the provider produced nothing usable — which is precisely the
    # case another provider can answer. Without this the campaign dropped to
    # the offline policy three times in eight steps.
    "RETURNED NO CHOICES",
    "NO CHOICES",
)

#: Not retryable even when the text also matches above.
_FATAL_MARKERS = ("401", "403", "UNAUTHORIZED", "INVALID API KEY", "400", "404")


def _is_retryable_upstream(exc: BaseException) -> bool:
    text = str(exc).upper()
    if any(marker in text for marker in _FATAL_MARKERS):
        return False
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _cooldown_seconds(exc: BaseException) -> float:
    """How long to shelve a provider. Honours Retry-After when present."""
    match = re.search(r"RETRY[- ]?AFTER[\"\':\s]+(\d+)", str(exc), re.IGNORECASE)
    if match:
        try:
            return min(300.0, float(match.group(1)))
        except ValueError:
            pass
    return 60.0


class RoleBoundClient:
    """``complete(system, user)`` for one role. Satisfies labpilot's LLMClient.

    Resolution happens **per call**, not once at construction: campaigns run for
    hours and a provider's daily cap is reached mid-run, so a client bound to a
    provider at startup would hold a dead one until the process exits.
    """

    def __init__(self, gateway: LLMGateway, role: str) -> None:
        self._gateway = gateway
        self.role = role
        self.last_served: ServedBy | None = None

    @property
    def model(self) -> str:
        """Best-effort current model. Some callers log this."""
        decision = self._gateway.preview(self.role)
        return decision.model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Complete for this role, failing over to the next provider if needed.

        Selection is *predictive*: it knows what our own ledger says we have
        spent, not what the upstream thinks. Those disagree — an OpenRouter
        model returned 429 on 2026-08-07 while our accounting showed budget
        left, because the limit was the upstream's, not ours. With nine
        providers configured and no failover, that single response ended a
        campaign.

        A failed provider is cooled down rather than skipped ad hoc, so the
        ordinary `select_route` walk does the work and the next call can use it
        again the moment the cooldown expires. `BudgetLedger.cool_down` was
        written for this and had never been called.
        """
        attempts = max(1, int(getattr(self._gateway.routing, "max_failover_attempts", 4)))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                # Waiting is legitimate on the first attempt — "everything is
                # rate-limited, the window reopens in 20s" is worth pacing for.
                # On a failover attempt it is not: we are here because a
                # provider just failed and we cooled it down ourselves, so
                # sleeping that cooldown out inside one call trades a fast, and
                # actionable, error for a stalled campaign.
                return self._complete_once(
                    system, user, json_mode=json_mode, allow_wait=attempt == 1
                )
            except RoleUnavailable:
                # Everything we could reach has now been cooled down by this
                # very loop. "no provider available" is true but useless — the
                # informative answer is why they failed, and the failure
                # taxonomy downstream keys on that message. Reporting
                # RoleUnavailable here would bucket a chain of 429s as `other`.
                if last_exc is None:
                    raise
                raise last_exc from None
            except _ProviderCallFailed as failure:
                last_exc = failure.original
                if attempt >= attempts or not _is_retryable_upstream(failure.original):
                    raise failure.original from None
                self._gateway.ledger.cool_down(
                    failure.provider,
                    _cooldown_seconds(failure.original),
                    reason=type(failure.original).__name__,
                )
                logger.warning(
                    "role=%s provider %s failed (%s); failing over (attempt %d/%d)",
                    self.role,
                    failure.provider,
                    failure.original,
                    attempt,
                    attempts,
                )
        assert last_exc is not None
        raise last_exc

    def _complete_once(
        self, system: str, user: str, *, json_mode: bool = False, allow_wait: bool = True
    ) -> str:
        gateway = self._gateway
        spec = gateway.routing.role_spec(self.role)

        decision = select_route(
            gateway.routing,
            self.role,
            gateway.ledger,
            credential_resolver=gateway.credential_resolver,
        )
        if decision.provider is None:
            if (
                not allow_wait
                or decision.wait_seconds <= 0
                or decision.wait_seconds > spec.max_wait_seconds
            ):
                raise RoleUnavailable(
                    f"role {self.role!r}: {decision.reason}"
                    + (
                        f" (would wait {decision.wait_seconds:.0f}s, "
                        f"over the {spec.max_wait_seconds:.0f}s bound)"
                        if decision.wait_seconds > 0
                        else ""
                    )
                )
            logger.info(
                "role=%s waiting %.0fs (%s)", self.role, decision.wait_seconds, decision.reason
            )
            time.sleep(decision.wait_seconds)
            decision = select_route(
                gateway.routing,
                self.role,
                gateway.ledger,
                credential_resolver=gateway.credential_resolver,
            )
            if decision.provider is None:
                raise RoleUnavailable(f"role {self.role!r} still unavailable: {decision.reason}")

        provider = decision.provider
        model = decision.model

        key = None
        if gateway.cache is not None:
            # The model is part of the key: same prompt, different provider,
            # different output — reproducing a stored result requires pinning
            # what produced it.
            key = cache_key(f"{provider.name}/{model}", user, gateway.temperature, system)
            cached = gateway.cache.get(key)
            if cached is not None:
                self.last_served = ServedBy(
                    provider=provider.name,
                    model=model,
                    tier=provider.tier,
                    role=self.role,
                    degraded=decision.degraded,
                    cache_hit=True,
                )
                logger.debug("role=%s cache hit on %s", self.role, self.last_served)
                return cached

        adapter = build_adapter(
            provider.kind,
            base_url=provider.base_url,
            api_key=gateway._api_key(provider.api_key_env),
            timeout_seconds=provider.request_timeout_seconds,
        )

        started = time.monotonic()
        try:
            result = adapter.complete(
                system,
                user,
                model=model,
                temperature=gateway.temperature,
                json_mode=json_mode,
            )
        except Exception as exc:
            # A failed call still consumed quota on most providers, so it is
            # recorded before the exception propagates.
            gateway.ledger.record(provider.name)
            raise _ProviderCallFailed(provider.name, exc) from exc
        latency_ms = int((time.monotonic() - started) * 1000)

        gateway.ledger.record(provider.name, tokens=result.total_tokens)
        self.last_served = ServedBy(
            provider=provider.name,
            model=model,
            tier=provider.tier,
            role=self.role,
            degraded=decision.degraded,
            tokens=result.total_tokens if result.metered else None,
            latency_ms=latency_ms,
        )
        logger.info("role=%s served by %s in %dms", self.role, self.last_served, latency_ms)

        if gateway.cache is not None and key is not None:
            gateway.cache.set(key, result.text, model=f"{provider.name}/{model}")
        return result.text
