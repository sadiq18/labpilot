"""fitroute — budget-aware, capability-checked LLM routing.

A caller declares *what kind of work* it is doing; the router returns the best
model that can actually do it, within entitlement, data policy and budget::

    gateway = LLMGateway(routing, ledger, cache=cache)
    client = gateway.for_role("codegen")
    text = client.complete(system, user, json_mode=True)
    print(client.last_served)      # groq/llama-3.3-70b · free

This package deliberately imports nothing from ``labpilot``. It lives in this
repository for now so its first consumer can exercise it, and is intended to be
extracted to a standalone open-source package — at which point extraction is a
directory move. ``tests/unit/test_fitroute_boundary.py`` enforces that.

Design: ``docs/smart-router/DESIGN.md``. This is v0.1 — the outcome-learning
bandit, discovery, streaming and ``role="auto"`` described there are future
work and deliberately absent.
"""

from fitroute.adapters import Completion, OllamaAdapter, OpenAICompatAdapter, build_adapter
from fitroute.budget import Availability, BudgetLedger
from fitroute.cache import PromptCache, cache_key
from fitroute.catalog import (
    ProviderSpec,
    RoleSpec,
    RoutingConfig,
    allowed_tiers,
    eligible_providers,
)
from fitroute.gateway import LLMGateway, RoleBoundClient, RoleUnavailable, ServedBy
from fitroute.select import RouteDecision, select_route

__all__ = [
    "Availability",
    "BudgetLedger",
    "Completion",
    "LLMGateway",
    "OllamaAdapter",
    "OpenAICompatAdapter",
    "PromptCache",
    "ProviderSpec",
    "RoleBoundClient",
    "RoleSpec",
    "RoleUnavailable",
    "RouteDecision",
    "RoutingConfig",
    "ServedBy",
    "allowed_tiers",
    "build_adapter",
    "cache_key",
    "eligible_providers",
    "select_route",
]
