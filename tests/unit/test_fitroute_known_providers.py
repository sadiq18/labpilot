"""The shipped provider catalog, and how a deployment selects from it.

The boundary these tests defend: fitroute owns *vendor facts* (endpoints, model
names, limits), a deployment owns *the choice* (which, in what order), and the
workspace owns the keys. Putting the catalog in labpilot coupled the framework
to a specific set of vendors, which is why it moved here.
"""

from __future__ import annotations

import pytest

from fitroute.catalog import (
    ProviderSpec,
    RoleSpec,
    RoutingConfig,
    eligible_providers,
    known_providers,
)


def test_catalog_loads_and_every_entry_is_usable():
    catalog = known_providers()
    assert catalog, "shipped catalog must not be empty"
    for name, spec in catalog.items():
        assert spec.name == name, "name is taken from the key, not repeated in the body"
        assert spec.models, f"{name} declares no model and could never serve a role"
        assert spec.note, f"{name} has no provenance note — a limit with no source is a guess"


def test_catalog_contains_no_credentials():
    """A shipped data file naming an env var is fine; one holding a key is not.

    So this checks credential *values*, not the substring ``api_key`` — which
    appears legitimately in ``api_key_env`` and made an earlier version of this
    test fail on correct data.
    """
    from fitroute.catalog import _KNOWN_PROVIDERS_FILE

    blob = _KNOWN_PROVIDERS_FILE.read_text(encoding="utf-8")
    for prefix in ("sk-", "gsk_", "AIza", "github_pat_", "Bearer "):
        assert prefix not in blob, f"catalog appears to contain a credential ({prefix})"

    for name, spec in known_providers().items():
        # Must be the *name* of a variable, never its contents.
        assert spec.api_key_env == spec.api_key_env.upper(), (
            f"{name}.api_key_env should be an env var name, got {spec.api_key_env!r}"
        )


def test_nothing_is_enabled_without_use():
    """Installing the router must not change which model answers.

    Auto-enabling the catalog would also add a `research doctor` row per role in
    every workspace, which is how the diagnostics tests caught this.
    """
    assert RoutingConfig().providers == []
    assert RoutingConfig(plan="free").providers == []


def test_use_expands_in_preference_order():
    routing = RoutingConfig(use=["ollama-local", "groq-llama70b"])
    assert [p.name for p in routing.providers] == ["ollama-local", "groq-llama70b"]
    # Order is the operator's statement of preference and must survive verbatim —
    # there is deliberately no tier ranking that would reorder local after cloud.
    assert routing.providers[0].tier == "local"


def test_unknown_name_fails_loudly():
    """A dropped name means a provider the operator believes is configured is
    never tried, surfacing much later as the far vaguer 'no eligible provider'."""
    with pytest.raises(ValueError, match="unknown provider"):
        RoutingConfig(use=["groq-llama70b", "gorq-typo"])


def test_inline_entry_overrides_a_catalog_entry_in_place():
    """A deployment adjusting one field must not lose its position in the order."""
    routing = RoutingConfig(
        use=["groq-llama70b", "ollama-local"],
        providers=[ProviderSpec(name="ollama-local", tier="local", models={"default": "mine"})],
    )
    assert [p.name for p in routing.providers] == ["groq-llama70b", "ollama-local"]
    assert routing.providers[1].model_for("default") == "mine"


def test_inline_entry_with_a_new_name_is_appended():
    """Bring-your-own-inference: an endpoint the shipped catalog cannot know."""
    routing = RoutingConfig(
        use=["groq-llama70b"],
        providers=[
            ProviderSpec(name="my-vllm", base_url="http://10.0.0.2:8000/v1", tier="local")
        ],
    )
    assert [p.name for p in routing.providers] == ["groq-llama70b", "my-vllm"]


def test_role_split_keeps_the_strong_daily_budget_for_strong_work():
    """The measured reason the catalog is shaped this way.

    groq-llama70b allows 1k requests/day and declares models for codegen and
    reasoning only; groq-llama8b allows 14.4k/day and serves everything else. If
    the 70b entry gained a `default` model it would also serve summarisation and
    spend the strong budget on it — the regression this test exists to catch.
    """
    routing = RoutingConfig(
        use=["groq-llama70b", "groq-llama8b"],
        roles={
            "codegen": RoleSpec(requires_strong=True, requires={"structured_output"}),
            "summarize": RoleSpec(requires={"structured_output"}),
        },
    )
    resolver = lambda _name: "key"  # noqa: E731

    codegen = eligible_providers(routing, "codegen", credential_resolver=resolver)
    summarize = eligible_providers(routing, "summarize", credential_resolver=resolver)

    assert [p.name for p in codegen] == ["groq-llama70b"]
    assert [p.name for p in summarize] == ["groq-llama8b"]


def test_codegen_entry_has_headroom_for_the_measured_prompt():
    """TPM binds codegen, not RPM: the rendered codegen prompt measures ~6.4k
    tokens, so the entry leading the strong chain needs room for a reply."""
    catalog = known_providers()
    assert catalog["groq-llama70b"].tpm >= 12000


# --- role keys must be roles this system actually has -----------------------

#: The only roles `model_for` will ever be asked for. A `models` key outside
#: this set is silently ignored — `model_for` returns None and
#: `eligible_providers` drops the entry — so the provider looks configured and
#: never serves anything.
KNOWN_ROLES = frozenset({"codegen", "reasoning", "summarize", "default"})


def test_no_entry_declares_a_role_this_system_does_not_have():
    """Measured 2026-08-07: two entries shipped declaring `coding` and
    `intelligence`. One served only reasoning (codegen silently lost), the other
    served nothing at all. Neither raised."""
    offenders = {
        name: sorted(set(spec.models) - KNOWN_ROLES)
        for name, spec in known_providers().items()
        if set(spec.models) - KNOWN_ROLES
    }
    assert not offenders, (
        f"unknown role keys — these entries are inert: {offenders}. "
        f"Valid roles: {sorted(KNOWN_ROLES)}"
    )


def test_every_entry_serves_at_least_one_role():
    """The stronger check: a spec can have valid-looking keys and still resolve
    to nothing for every role."""
    dead = [
        name
        for name, spec in known_providers().items()
        if not any(spec.model_for(role) for role in KNOWN_ROLES)
    ]
    assert not dead, f"entries that serve no role at all: {dead}"
