"""Provider keys must resolve from a workspace `.env`, whatever they are named.

`ProviderSpec.api_key_env` is config-driven, so the resolver has to handle names
`Settings` has never heard of. It previously read `Settings` fields only, which
silently dropped every key outside its fixed field list — including the
`GROQ_API_KEY` used by the example in `configs/default.yaml`.
"""

from __future__ import annotations

import pytest

from fitroute.catalog import ProviderSpec, RoleSpec, RoutingConfig, eligible_providers
from labpilot.config import Settings
from labpilot.llm.client import settings_credential_resolver


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    """A workspace `.env` holding a key that is not a Settings field."""
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=gsk-test-in-dotenv-only\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("labpilot.config.resolve_env_files", lambda: (str(env_file),))
    monkeypatch.setattr("labpilot.llm.client.Settings", lambda: Settings(_env_file=str(env_file)))
    return env_file


def _groq() -> ProviderSpec:
    return ProviderSpec(
        name="groq",
        api_key_env="GROQ_API_KEY",
        tier="free",
        strong=True,
        caps={"structured_output"},
        models={"default": "llama-3.3-70b-versatile"},
    )


def test_settings_field_key_still_resolves(tmp_path, monkeypatch):
    """The path that already worked must keep working."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    monkeypatch.setattr("labpilot.config.resolve_env_files", lambda: (str(env_file),))

    resolve = settings_credential_resolver(Settings(_env_file=str(env_file)))
    assert resolve("OPENAI_API_KEY") == "sk-test"


def test_non_settings_key_resolves_from_dotenv(workspace_env):
    """The reported bug: Settings uses extra="ignore", so GROQ_API_KEY never
    becomes an attribute and the resolver returned empty."""
    resolve = settings_credential_resolver()
    assert resolve("GROQ_API_KEY") == "gsk-test-in-dotenv-only"


def test_provider_with_dotenv_only_key_is_eligible(workspace_env):
    """The user-visible symptom: the documented Groq example produced
    "no eligible provider" with the key sitting in the file just edited."""
    routing = RoutingConfig(
        plan="free",
        providers=[_groq()],
        roles={"codegen": RoleSpec(requires_strong=True, requires={"structured_output"})},
    )
    assert eligible_providers(routing, "codegen") == []

    found = eligible_providers(
        routing, "codegen", credential_resolver=settings_credential_resolver()
    )
    assert [p.name for p in found] == ["groq"]


def test_missing_key_still_resolves_empty(workspace_env):
    resolve = settings_credential_resolver()
    assert resolve("NOT_IN_ANY_FILE") == ""


def test_unreadable_env_file_does_not_raise(monkeypatch, tmp_path):
    """Credential lookup runs on every routing decision; it must never throw."""
    monkeypatch.setattr(
        "labpilot.config.resolve_env_files", lambda: (str(tmp_path / "nope" / ".env"),)
    )
    assert settings_credential_resolver(Settings(_env_file=None))("ANY_KEY") == ""
