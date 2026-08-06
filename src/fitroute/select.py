"""The routing decision — a pure function.

No clock, no network, no filesystem. Availability and budget are computed by
the caller and passed in, so every routing property (entitlement, data policy,
capability, budget, wait-vs-degrade) is a table-driven unit test.

That purity is not tidiness. A router that constructs clients cannot be tested
without a network, which is why routing logic is untested nearly everywhere it
exists — and it is what keeps this module portable to another language later.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitroute.budget import BudgetLedger
from fitroute.catalog import (
    CredentialResolver,
    ProviderSpec,
    RoutingConfig,
    allowed_tiers,
    eligible_providers,
)


@dataclass(frozen=True)
class RouteDecision:
    """Outcome of routing.

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
    credential_resolver: CredentialResolver | None = None,
) -> RouteDecision:
    """Pick a provider for ``role`` within entitlement, policy and budget.

    Exhaustion is handled per role rather than globally. Downgrading a codegen
    or hypothesis call to a weak model is worse than waiting: the weak output
    gets recorded as "this technique did not help", a false negative that is
    indistinguishable from a real one and permanently pollutes research memory.
    Summarisation has no such property and degrades freely.
    """
    spec = routing.role_spec(role)
    candidates = eligible_providers(routing, role, credential_resolver=credential_resolver)

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
        best_reason = _no_candidate_reason(routing, role, credential_resolver)

    if spec.on_exhaustion == "degrade":
        # Strength is relaxed; required capabilities never are.
        for provider in eligible_providers(
            routing, role, ignore_strength=True, credential_resolver=credential_resolver
        ):
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

    # `fail` means raise now, so no wait is reported — a caller that paces on
    # wait_seconds would otherwise sleep first and make `fail` behave like
    # `wait`.
    wait = 0.0 if spec.on_exhaustion == "fail" else (best_wait or 0.0)
    return RouteDecision(provider=None, role=role, wait_seconds=wait, reason=best_reason)


def _no_candidate_reason(
    routing: RoutingConfig,
    role: str,
    resolver: CredentialResolver | None,
) -> str:
    """Say *which* filter emptied the list.

    "No eligible provider" with no cause is the failure mode that sends people
    to read the router's source. Naming the filter that rejected everything is
    the difference between a two-minute fix and an afternoon.
    """
    spec = routing.role_spec(role)
    if not routing.providers:
        return "no providers configured"

    allowed = allowed_tiers(routing.plan)
    surviving = [p for p in routing.providers if p.tier in allowed]
    if not surviving:
        return f"plan {routing.plan!r} permits only {sorted(allowed)} and no provider is in them"

    surviving = [p for p in surviving if routing.allow_training_on_inputs or not p.trains_on_input]
    if not surviving:
        return "all providers may train on inputs and allow_training_on_inputs is false"

    with_creds = [p for p in surviving if p.has_credentials(resolver)]
    if not with_creds:
        missing = sorted({p.api_key_env for p in surviving if p.api_key_env})
        return f"no credentials found (set one of: {', '.join(missing) or 'n/a'})"

    with_model = [p for p in with_creds if p.model_for(role) is not None]
    if not with_model:
        return f"no provider declares a model for role {role!r}"

    capable = [p for p in with_model if spec.requires <= p.caps]
    if not capable:
        wanted = sorted(spec.requires)
        return f"no provider supports required capabilities {wanted} for role {role!r}"

    if spec.requires_strong:
        return f"role {role!r} requires a strong provider and none is configured"
    return "no eligible provider"
