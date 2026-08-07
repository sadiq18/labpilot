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

import json
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

#: Vendor facts shipped with the router: endpoints, models and measured limits.
#: Deliberately *not* enabled by default — a deployment names what it wants via
#: ``RoutingConfig.use``. Auto-enabling would make merely installing the router
#: change which model answers, and would put vendor choices in the hands of the
#: package rather than the operator.
_KNOWN_PROVIDERS_FILE = Path(__file__).with_name("known_providers.json")

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
    # How the numbers above were arrived at — measured, published, or unverified.
    # Carried with the spec so a stale or guessed limit is visible at the point
    # someone is deciding whether to trust it.
    note: str = ""

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


#: Capabilities every role requires, whatever a deployment writes in `requires`.
#: See `RoleSpec.requires` for why this one cannot be relaxed.
MANDATORY_CAPS: frozenset[str] = frozenset({"structured_output"})


class RoleSpec(BaseModel):
    """What a class of work needs, and what to do when it cannot be had."""

    requires_strong: bool = False
    # Capabilities a provider must have to serve this role at all. A hard
    # precondition, checked before any ranking: routing a JSON-parsing role to
    # a model that cannot produce JSON is not a degraded result, it is a
    # guaranteed one.
    # `MANDATORY_CAPS` is unioned in by the validator below and cannot be
    # dropped: with the rule engines deleted (M14 phase 3) nothing else catches
    # a model that answers a JSON-only prompt in prose.
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

    @model_validator(mode="after")
    def _enforce_mandatory_caps(self) -> RoleSpec:
        """Union in any mandatory capability the config left out or removed.

        Restoring silently beats raising: omitting `requires` entirely is the
        common, correct case, and the guarantee should hold regardless of what
        a deployment wrote. Measured on rogii 2026-08-07, the prose-reply
        failure is unreachable while `structured_output` is required — which is
        what makes deleting the deterministic fallbacks safe. A workspace that
        relaxed it would quietly reintroduce the failure to a system that no
        longer has a net.
        """
        missing = MANDATORY_CAPS - self.requires
        if missing:
            self.requires = set(self.requires) | missing
        return self


@lru_cache(maxsize=1)
def known_providers() -> dict[str, ProviderSpec]:
    """Endpoints the router ships knowledge of, by name.

    Data, not policy: nothing here is active until a deployment names it in
    ``RoutingConfig.use``. Keys never appear in this file — only the *name* of
    the environment variable each endpoint reads.
    """
    raw = json.loads(_KNOWN_PROVIDERS_FILE.read_text(encoding="utf-8"))
    return {
        name: ProviderSpec(name=name, **fields)
        for name, fields in raw.get("providers", {}).items()
    }

class RoutingConfig(BaseModel):
    """Catalog + entitlement + per-role requirements."""

    plan: str = "free"
    # Hard override, independent of plan: refuse any endpoint that may train on
    # submitted content. Workspaces holding proprietary data set this False.
    allow_training_on_inputs: bool = True
    # Names from the shipped catalog, **in preference order**. This is how a
    # deployment states "these providers, tried in this sequence" without
    # restating vendor facts it does not own.
    use: list[str] = Field(default_factory=list)
    # Inline definitions: custom or self-hosted endpoints the shipped catalog
    # cannot know about. An inline entry whose name matches a `use` entry
    # replaces it *in place*, so a deployment can adjust one field of a known
    # provider without losing its position in the preference order.
    providers: list[ProviderSpec] = Field(default_factory=list)
    roles: dict[str, RoleSpec] = Field(default_factory=dict)
    # How many providers to try before giving up on one call. Selection is
    # predictive — it knows our ledger, not the upstream's — so a provider can
    # 429 while we believe it has budget. Bounded because failing over through a
    # nine-deep chain on a fatal error just reaches the same error slowly.
    max_failover_attempts: int = 4

    @model_validator(mode="after")
    def _expand_use(self) -> RoutingConfig:
        if not self.use:
            return self

        catalog = known_providers()
        unknown = [name for name in self.use if name not in catalog]
        if unknown:
            # Fail loudly. A silently dropped name means a provider the operator
            # believes is configured never gets tried, and the symptom shows up
            # much later as "no eligible provider".
            raise ValueError(
                f"unknown provider(s) in routing.use: {unknown}. "
                f"Available: {sorted(catalog)}"
            )

        overrides = {p.name: p for p in self.providers}
        resolved = [overrides.pop(name, catalog[name]) for name in self.use]
        # Inline entries not overriding a `use` name keep their relative order
        # and follow, so `use` states the primary preference.
        self.providers = resolved + [p for p in self.providers if p.name in overrides]
        return self

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
