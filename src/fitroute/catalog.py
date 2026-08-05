"""Provider catalog and entitlement rules for LLM routing.

Three concerns are deliberately kept apart, because conflating them is what
makes a router impossible to extend later:

* **Role requirement** — what the *task* needs (frontier reasoning vs cheap
  summarisation). Set by the code doing the work.
* **Provider catalog** — what models exist, their limits, whether they train on
  inputs, and what they cost. Pure data, lives in YAML, changes weekly.
* **Entitlement** — what *this user* may use. Driven by their plan, not by the
  task and not by the catalog.

Routing is the intersection of the three. Adding a paid tier is then a config
change plus a plan name, not a rewrite.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from pydantic import BaseModel, Field

#: Resolves an env var name to its value. Injected so this module stays free of
#: any ``labpilot`` import (router-core rule) while still seeing keys that live
#: only in a workspace ``.env`` — pydantic-settings loads those into a Settings
#: object and never exports them to ``os.environ``.
CredentialResolver = Callable[[str], str]

# Which provider tiers each plan may draw on. Enterprise deliberately excludes
# free tiers: their data policies generally permit training on inputs, which is
# not acceptable for a customer's proprietary research.
_PLAN_TIERS: dict[str, set[str]] = {
    "free": {"free", "local"},
    "pro": {"paid", "free", "local"},
    "enterprise": {"paid", "local"},
}


class ProviderSpec(BaseModel):
    """One routable endpoint. Pure data — no behaviour, no credentials."""

    name: str
    kind: str = "openai_compat"  # openai_compat | gemini | ollama
    base_url: str = ""
    api_key_env: str = ""
    tier: str = "free"  # free | paid | local
    # Whether this endpoint is fit for frontier-grade reasoning (conductor
    # policy, hypothesis generation, code generation).
    strong: bool = False
    # Free tiers commonly reserve the right to train on submitted content.
    trains_on_input: bool = False
    # role -> model name; "default" is the fallback for unlisted roles.
    models: dict[str, str] = Field(default_factory=dict)
    # What this endpoint can actually do. `structured_output` is the one that
    # matters most: a model that cannot constrain its output to JSON will
    # cheerfully answer a JSON-only prompt in prose, return HTTP 200, and have
    # the reply discarded downstream — observed as a 3-of-3 fallback rate before
    # constrained decoding was requested.
    caps: set[str] = Field(default_factory=set)
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    request_timeout_seconds: float = 600.0

    def model_for(self, role: str) -> str | None:
        return self.models.get(role) or self.models.get("default")

    def has_credentials(self, resolver: CredentialResolver | None = None) -> bool:
        """Local providers need none; everything else needs its key resolvable.

        ``resolver`` is consulted before the process environment so a key that
        lives only in a workspace ``.env`` is visible. Without it, routing
        reports "no eligible provider" while the key sits in the file the user
        just edited.
        """
        if self.tier == "local" or not self.api_key_env:
            return self.tier == "local"
        if resolver is not None and resolver(self.api_key_env).strip():
            return True
        return bool(os.environ.get(self.api_key_env, "").strip())


class RoleSpec(BaseModel):
    """What a class of work needs, and what to do when it cannot be had."""

    requires_strong: bool = False
    # Capabilities a provider must have to serve this role at all. A hard
    # precondition, checked before any ranking: routing a JSON-parsing role to
    # a model that cannot produce JSON is not a degraded result, it is a
    # guaranteed one.
    requires: set[str] = Field(default_factory=set)
    # wait   — queue until a capable provider frees up (default for reasoning
    #          and codegen: a weak model silently produces a *false negative*,
    #          recording "technique X failed" when really "the writer failed").
    # degrade— accept a weaker provider (fine for summarisation).
    # fail   — raise rather than proceed.
    on_exhaustion: str = "degrade"
    # Bounded, because an unbounded wait in an unattended campaign is
    # indistinguishable from a hang.
    max_wait_seconds: float = 900.0


class RoutingConfig(BaseModel):
    """Catalog + entitlement + per-role requirements."""

    plan: str = "free"
    # Hard override, independent of plan: refuse any endpoint that may train on
    # submitted content. Workspaces holding proprietary data set this False.
    allow_training_on_inputs: bool = True
    providers: list[ProviderSpec] = Field(default_factory=list)
    roles: dict[str, RoleSpec] = Field(default_factory=dict)

    def role_spec(self, role: str) -> RoleSpec:
        return self.roles.get(role) or self.roles.get("default") or RoleSpec()


def allowed_tiers(plan: str) -> set[str]:
    """Tiers a plan may draw on; unknown plans get the most restrictive set."""
    return _PLAN_TIERS.get((plan or "").strip().lower(), _PLAN_TIERS["free"])


def eligible_providers(
    routing: RoutingConfig,
    role: str,
    *,
    ignore_strength: bool = False,
    credential_resolver: CredentialResolver | None = None,
) -> list[ProviderSpec]:
    """Providers that may serve ``role``, in preference order.

    ``ignore_strength`` is used only by an explicit degrade step, so that
    relaxing capability is always a deliberate, recorded decision rather than a
    silent side effect of exhaustion. Required *capabilities* are never relaxed
    — degrading to a model that cannot do the job is not a degraded result.

    Order is **catalog order**. There is deliberately no tier ranking: a free
    local model that answers correctly for zero cost and no network should be
    able to outrank a paid one, and a hardcoded ``paid > free > local`` makes
    that impossible. Preference is the operator's to state, by listing
    providers in the order they want them tried.
    """
    tiers = allowed_tiers(routing.plan)
    spec = routing.role_spec(role)
    needs_strong = spec.requires_strong and not ignore_strength

    return [
        provider
        for provider in routing.providers
        if provider.tier in tiers
        and (routing.allow_training_on_inputs or not provider.trains_on_input)
        and (provider.strong or not needs_strong)
        and spec.requires <= provider.caps
        and provider.model_for(role) is not None
        and provider.has_credentials(credential_resolver)
    ]
