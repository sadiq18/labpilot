"""A 429 from one provider must not end a campaign that has eight more.

Route selection is *predictive*: it consults our own ledger, which knows what we
spent, not what the upstream thinks. Those disagree — an OpenRouter model
returned 429 on 2026-08-07 while our accounting showed budget remaining. With
nine providers configured, `RoleBoundClient.complete` recorded the failure and
re-raised, and the campaign stopped.

`BudgetLedger.cool_down` had been written for exactly this and was never called
from anywhere.
"""

from __future__ import annotations

import pytest

from fitroute.budget import BudgetLedger
from fitroute.catalog import ProviderSpec, RoleSpec, RoutingConfig
from fitroute.gateway import LLMGateway, _cooldown_seconds, _is_retryable_upstream


@pytest.mark.parametrize(
    ("message", "retryable"),
    [
        ("429 Too Many Requests", True),
        ("RESOURCE_EXHAUSTED", True),
        ("503 Service Unavailable", True),
        ("upstream overloaded", True),
        ("Connection reset by peer", True),
        ("Read timed out", True),
        # Failing over on these just reaches the same error more slowly.
        ("401 Unauthorized", False),
        ("403 Forbidden", False),
        ("400 Bad Request: unknown model", False),
        ("404 not found", False),
        ("invalid api key", False),
    ],
)
def test_only_upstream_conditions_fail_over(message, retryable):
    assert _is_retryable_upstream(Exception(message)) is retryable


def test_a_fatal_marker_wins_over_a_retryable_one():
    """A 400 whose body happens to mention a timeout is still fatal."""
    assert _is_retryable_upstream(Exception("400 Bad Request (timeout in prompt)")) is False


def test_retry_after_is_honoured():
    assert _cooldown_seconds(Exception("429, Retry-After: 30")) == 30.0
    assert _cooldown_seconds(Exception('429 {"retry-after": "12"}')) == 12.0


def test_cooldown_is_bounded_and_defaulted():
    assert _cooldown_seconds(Exception("503 unavailable")) == 60.0
    assert _cooldown_seconds(Exception("429 Retry-After: 99999")) == 300.0


# --- end to end through the gateway -----------------------------------------


def _routing() -> RoutingConfig:
    return RoutingConfig(
        plan="free",
        providers=[
            ProviderSpec(
                name=name,
                kind="openai_compat",
                base_url=f"https://{name}.test/v1",
                api_key_env="TEST_KEY",
                models={"default": f"{name}-model"},
                caps={"structured_output"},
                tier="free",
                rpm=100,
                rpd=1000,
            )
            for name in ("first", "second", "third")
        ],
        roles={"default": RoleSpec(requires=["structured_output"])},
    )


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    return LLMGateway(
        routing=_routing(),
        ledger=BudgetLedger(tmp_path / "ledger.sqlite"),
        cache=None,
    )


class _Adapters:
    """Stands in for build_adapter, scripted per provider by base_url."""

    def __init__(self, script: dict[str, object]) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(self, kind, *, base_url, api_key, timeout_seconds):
        name = base_url.split("//")[1].split(".")[0]
        outcome = self.script.get(name)
        calls = self.calls

        class _Adapter:
            def complete(self, system, user, *, model, temperature, json_mode):
                calls.append(name)
                if isinstance(outcome, Exception):
                    raise outcome

                class _R:
                    text = f"ok from {name}"
                    total_tokens = 10
                    metered = True

                return _R()

        return _Adapter()


def test_a_429_fails_over_to_the_next_provider(gateway, monkeypatch):
    adapters = _Adapters({"first": RuntimeError("429 rate limited"), "second": None})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)

    assert gateway.for_role("default").complete("s", "u") == "ok from second"
    assert adapters.calls == ["first", "second"]


def test_the_failed_provider_is_cooled_down_not_just_skipped(gateway, monkeypatch):
    """Cooling down means the *ordinary* selection walk avoids it, and picks it
    up again when the window passes — no ad hoc skip list."""
    adapters = _Adapters({"first": RuntimeError("429"), "second": None})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)
    gateway.for_role("default").complete("s", "u")

    avail = gateway.ledger.availability("first", rpm=100, rpd=1000, tpm=None)
    assert not avail.ok


def test_a_fatal_error_does_not_burn_the_chain(gateway, monkeypatch):
    adapters = _Adapters({"first": RuntimeError("401 Unauthorized"), "second": None})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)

    with pytest.raises(RuntimeError, match="401"):
        gateway.for_role("default").complete("s", "u")
    assert adapters.calls == ["first"], "must not try the rest on a bad key"


def test_the_original_exception_is_what_callers_see(gateway, monkeypatch):
    """`_is_transient_llm_error` and the provenance classifier both read the
    message, so the internal wrapper must never escape."""
    adapters = _Adapters({name: RuntimeError("429 everywhere") for name in ("first", "second", "third")})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)

    with pytest.raises(RuntimeError) as exc:
        gateway.for_role("default").complete("s", "u")
    assert str(exc.value) == "429 everywhere"
    assert type(exc.value) is RuntimeError


def test_failover_is_bounded(gateway, monkeypatch):
    """Every provider down must terminate, not loop."""
    adapters = _Adapters({name: RuntimeError("503") for name in ("first", "second", "third")})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)

    with pytest.raises(RuntimeError):
        gateway.for_role("default").complete("s", "u")
    assert len(adapters.calls) <= gateway.routing.max_failover_attempts


def test_a_working_provider_is_not_retried(gateway, monkeypatch):
    adapters = _Adapters({"first": None})
    monkeypatch.setattr("fitroute.gateway.build_adapter", adapters)

    assert gateway.for_role("default").complete("s", "u") == "ok from first"
    assert adapters.calls == ["first"]
