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
        gateway = self._gateway
        spec = gateway.routing.role_spec(self.role)

        decision = select_route(
            gateway.routing,
            self.role,
            gateway.ledger,
            credential_resolver=gateway.credential_resolver,
        )
        if decision.provider is None:
            if decision.wait_seconds <= 0 or decision.wait_seconds > spec.max_wait_seconds:
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
        except Exception:
            # A failed call still consumed quota on most providers, so it is
            # recorded before the exception propagates.
            gateway.ledger.record(provider.name)
            raise
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
