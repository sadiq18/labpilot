"""The proxy exists so an external tool cannot bypass the router.

Every test here is about that: a request that reaches a provider without going
through `select_route` has taken the ledger, the rate limits, the failover and
the provenance with it — which is the whole reason M19 needs this before the
aider adapter.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http import HTTPStatus

import pytest

from fitroute.budget import BudgetLedger
from fitroute.catalog import (
    CAPS_EXEMPT_ROLES,
    MANDATORY_CAPS,
    ProviderSpec,
    RoleSpec,
    RoutingConfig,
)
from fitroute.gateway import LLMGateway, RoleUnavailable
from fitroute.server import (
    ProxyError,
    ProxyServer,
    completion_response,
    handle_chat_completion,
    handle_models,
    role_from_model,
    split_messages,
)


def _routing(**roles) -> RoutingConfig:
    return RoutingConfig(
        plan="free",
        providers=[
            ProviderSpec(
                name="p1",
                kind="openai_compat",
                base_url="https://p1.test/v1",
                api_key_env="TEST_KEY",
                models={"default": "p1-model", "codegen": "p1-coder"},
                caps={"structured_output"},
                tier="free",
                rpm=100,
                rpd=1000,
            )
        ],
        roles=roles or {"default": RoleSpec(), "codegen": RoleSpec()},
    )


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    return LLMGateway(routing=_routing(), ledger=BudgetLedger(tmp_path / "b.sqlite"), cache=None)


# --- roles must survive the HTTP boundary -----------------------------------


@pytest.mark.parametrize(
    ("model", "role"),
    [("labpilot/codegen", "codegen"), ("labpilot/default", "default"),
     ("LABPILOT/Codegen", "codegen")],
)
def test_a_role_model_resolves(model, role):
    """`labpilot/<role>` is how a role crosses HTTP."""
    assert role_from_model(model) == role


@pytest.mark.parametrize(
    "model",
    # `codegen` bare is in this list deliberately: an earlier version accepted
    # it "because litellm sometimes strips prefixes", which made `gpt-4o` a
    # valid role that then fell through to `default` — the bypass this function
    # exists to prevent, created by loosening the one token carrying the meaning.
    ["gpt-4o", "openai/gpt-4o", "nvidia/nemotron-3-super", "", "a/b/c", "codegen"],
)
def test_naming_a_provider_model_is_refused(model):
    """The failure this server exists to prevent, arriving through the server.

    Accepting a provider model would route it straight through and silently skip
    role selection — no per-role `requires`, no `on_exhaustion`, no limits.
    """
    with pytest.raises(ProxyError) as exc:
        role_from_model(model)
    assert exc.value.status == HTTPStatus.BAD_REQUEST
    assert "labpilot/<role>" in exc.value.message


# --- streaming is refused, not silently ignored -----------------------------


def test_streaming_is_refused(gateway):
    """Answering a stream request with a complete body would look like success
    while disabling the accounting this server exists to provide."""
    with pytest.raises(ProxyError) as exc:
        handle_chat_completion(gateway, {"model": "labpilot/default", "stream": True,
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.status == HTTPStatus.BAD_REQUEST
    assert "--no-stream" in exc.value.message


# --- message flattening ------------------------------------------------------


def test_system_and_user_are_separated():
    system, user = split_messages([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ])
    assert system == "be terse"
    assert user == "hello"


def test_multi_turn_is_joined_not_truncated():
    """aider sends the whole conversation; dropping turns silently changes the
    request into a different one."""
    _, user = split_messages([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ])
    assert "first" in user and "reply" in user and "second" in user


def test_content_parts_keep_their_text():
    _, user = split_messages([
        {"role": "user", "content": [{"type": "text", "text": "abc"}, {"type": "image_url"}]},
    ])
    assert user == "abc"


def test_an_empty_request_is_refused(gateway):
    with pytest.raises(ProxyError) as exc:
        handle_chat_completion(gateway, {"model": "labpilot/default", "messages": []})
    assert exc.value.status == HTTPStatus.BAD_REQUEST


# --- the response carries what the ledger recorded --------------------------


def test_usage_reports_recorded_tokens_not_an_invention():
    """Clients estimate cost from `usage`. A made-up number makes their
    accounting disagree with ours in a way neither side can detect."""
    from types import SimpleNamespace

    served = SimpleNamespace(provider="p1", model="p1-model", role="codegen",
                             tokens=42, degraded=False, cache_hit=False)
    body = completion_response("out", "labpilot/codegen", served)
    assert body["usage"]["total_tokens"] == 42
    assert body["choices"][0]["message"]["content"] == "out"


def test_the_response_names_who_served_it():
    """Same attribution the evidence card gets — a client logging the response
    can tie a result to the provider that produced it."""
    from types import SimpleNamespace

    served = SimpleNamespace(provider="p1", model="p1-coder", role="codegen",
                             tokens=1, degraded=True, cache_hit=False)
    body = completion_response("x", "labpilot/codegen", served)
    assert body["x_fitroute"]["provider"] == "p1"
    assert body["x_fitroute"]["degraded"] is True


def test_a_missing_served_stamp_does_not_crash():
    body = completion_response("x", "labpilot/default", None)
    assert body["usage"]["total_tokens"] == 0


# --- exhaustion answers 429, it does not hold the connection ----------------


def test_exhaustion_is_429_with_retry_after(gateway, monkeypatch):
    """Holding aider's connection open invites a client-side timeout, which
    becomes a retry and makes the pressure worse. Speak the protocol instead."""
    def _unavailable(self, *a, **k):
        raise RoleUnavailable("role 'default': everything is rate limited")

    monkeypatch.setattr("fitroute.gateway.RoleBoundClient.complete", _unavailable)
    with pytest.raises(ProxyError) as exc:
        handle_chat_completion(gateway, {"model": "labpilot/default",
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.status == HTTPStatus.TOO_MANY_REQUESTS
    assert exc.value.retry_after and exc.value.retry_after <= 300


# --- the model list advertises roles ----------------------------------------


def test_models_advertises_roles_not_providers(gateway):
    ids = {m["id"] for m in handle_models(gateway)["data"]}
    assert "labpilot/codegen" in ids
    assert not any("p1" in i for i in ids), "a provider name here would invite bypassing roles"


# --- the codegen carve-out --------------------------------------------------


def test_codegen_is_exempt_from_structured_output():
    """aider needs a good editor, not JSON mode. Requiring `structured_output`
    would exclude models that are excellent at editing for a capability the work
    never uses."""
    routing = _routing()
    assert not (MANDATORY_CAPS & routing.role_spec("codegen").requires)


@pytest.mark.parametrize("role", ["default", "reasoning", "summarize"])
def test_every_other_role_keeps_the_mandate(role):
    """Relaxing these reopens the prose-reply failure M14 phase 3 removed the
    net for."""
    assert MANDATORY_CAPS <= _routing().role_spec(role).requires


def test_the_exemption_does_not_mutate_the_shared_default():
    """`role_spec` falls back to the `default` spec, so exempting in place would
    silently strip the mandate from every role that falls through."""
    routing = _routing()
    routing.role_spec("codegen")
    assert MANDATORY_CAPS <= routing.role_spec("default").requires


def test_the_exemption_is_narrow():
    assert CAPS_EXEMPT_ROLES == frozenset({"codegen"})


# --- end to end over real HTTP ----------------------------------------------


def test_a_real_request_routes_through_the_gateway(gateway, monkeypatch):
    """The point of the whole module: an HTTP client reaches a provider only by
    way of `select_route`."""
    seen = {}

    def _adapter(kind, *, base_url, api_key, timeout_seconds):
        class _A:
            def complete(self, system, user, *, model, temperature, json_mode):
                seen["model"] = model
                seen["user"] = user

                class _R:
                    text = "routed"
                    total_tokens = 7
                    metered = True

                return _R()

        return _A()

    monkeypatch.setattr("fitroute.gateway.build_adapter", _adapter)
    with ProxyServer(gateway) as proxy:
        req = urllib.request.Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({
                "model": "labpilot/codegen",
                "messages": [{"role": "user", "content": "edit this"}],
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(req, timeout=10).read())

    assert body["choices"][0]["message"]["content"] == "routed"
    assert seen["model"] == "p1-coder", "the codegen role's model, chosen by the router"
    assert body["x_fitroute"]["provider"] == "p1"


def test_a_bad_model_is_rejected_over_http(gateway):
    with ProxyServer(gateway) as proxy:
        req = urllib.request.Request(
            f"{proxy.base_url}/chat/completions",
            data=json.dumps({"model": "gpt-4o",
                             "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=10)
        assert exc.value.code == 400


def test_the_port_is_ephemeral_and_released(gateway):
    """Bound to port 0 and closed by its owner, so concurrent campaigns cannot
    collide and an orphan cannot outlive the ledger it writes to."""
    with ProxyServer(gateway) as proxy:
        first = proxy.port
        assert first > 0
        assert proxy.base_url.endswith(f":{first}/v1")
    with ProxyServer(gateway) as second:
        assert second.port > 0


def test_an_explicit_requirement_outranks_the_exemption():
    """The exemption removes the *mandate*, not the config's own decision.

    A first version stripped `structured_output` from codegen unconditionally,
    which also discarded an explicit `requires={"structured_output"}` — and
    broke `test_model_without_structured_output_is_not_a_candidate`, whose whole
    subject is a codegen role that asked for it. `model_fields_set` cannot tell
    the two apart, because the validator's own assignment marks the field set;
    the validator records what it added instead.
    """
    routing = RoutingConfig(
        plan="free",
        providers=[],
        roles={"codegen": RoleSpec(requires={"structured_output"})},
    )
    assert routing.role_spec("codegen").requires == {"structured_output"}


def test_the_exemption_keeps_other_capabilities():
    routing = RoutingConfig(
        plan="free", providers=[], roles={"codegen": RoleSpec(requires={"vision"})}
    )
    assert routing.role_spec("codegen").requires == {"vision"}


# --- the blockers PR #110 review caught -------------------------------------


def test_the_shipped_config_actually_enables_the_carve_out():
    """The gap between a unit fixture and production.

    Every carve-out test above builds `RoleSpec()` with no explicit `requires`,
    so they assert the exemption while `configs/default.yaml` had
    `requires: [structured_output]` on codegen — an explicit requirement, which
    outranks the exemption by design. The feature was correct and disabled.
    """
    from pathlib import Path

    from labpilot.cli.config_helpers import load_cli_config

    config, _ = load_cli_config(config_path=Path("configs/default.yaml"))
    routing = config.llm.routing
    assert not (MANDATORY_CAPS & routing.role_spec("codegen").requires)


@pytest.mark.parametrize("role", ["default", "reasoning", "summarize"])
def test_the_shipped_config_keeps_the_mandate_elsewhere(role):
    from pathlib import Path

    from labpilot.cli.config_helpers import load_cli_config

    config, _ = load_cli_config(config_path=Path("configs/default.yaml"))
    assert MANDATORY_CAPS <= config.llm.routing.role_spec(role).requires


def test_the_proxy_never_sleeps_on_a_paced_wait(gateway, monkeypatch):
    """`complete` defaults to `allow_wait=True`, and codegen's `on_exhaustion`
    is `wait` with `max_wait_seconds: 900`.

    The call runs inside `_GATEWAY_LOCK`, so one paced wait would block every
    other proxied request for up to fifteen minutes. The earlier test mocked
    `complete` to raise immediately and never exercised this at all.
    """
    seen = {}

    def _capture(self, system, user, *, json_mode=False, allow_wait=True):
        seen["allow_wait"] = allow_wait
        raise RoleUnavailable("rate limited", retry_after=42)

    monkeypatch.setattr("fitroute.gateway.RoleBoundClient.complete", _capture)
    with pytest.raises(ProxyError):
        handle_chat_completion(gateway, {"model": "labpilot/codegen",
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert seen["allow_wait"] is False, "the proxy must refuse fast, not sleep under the lock"


def test_retry_after_reports_the_real_window(gateway, monkeypatch):
    """A constant tells the client to back off for the wrong length of time."""
    def _unavailable(self, *a, **k):
        raise RoleUnavailable("rate limited", retry_after=42)

    monkeypatch.setattr("fitroute.gateway.RoleBoundClient.complete", _unavailable)
    with pytest.raises(ProxyError) as exc:
        handle_chat_completion(gateway, {"model": "labpilot/default",
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.retry_after == 42


def test_retry_after_is_bounded(gateway, monkeypatch):
    def _unavailable(self, *a, **k):
        raise RoleUnavailable("rate limited", retry_after=99999)

    monkeypatch.setattr("fitroute.gateway.RoleBoundClient.complete", _unavailable)
    with pytest.raises(ProxyError) as exc:
        handle_chat_completion(gateway, {"model": "labpilot/default",
                                         "messages": [{"role": "user", "content": "hi"}]})
    assert exc.value.retry_after == 300


def test_in_process_callers_still_get_to_wait():
    """`allow_wait=False` is the proxy's choice, not a global behaviour change:
    "everything is limited, the window reopens in 20s" is worth pacing for when
    nobody else is blocked behind you."""
    import inspect

    from fitroute.gateway import RoleBoundClient

    assert inspect.signature(RoleBoundClient.complete).parameters["allow_wait"].default is True


# --- the ledger is safe on its own, not by convention -----------------------


def test_the_ledger_serialises_its_own_writes(tmp_path):
    """`check_same_thread=False` makes cross-thread use possible; only the lock
    makes it safe. Putting the lock in the caller means the rule lapses the
    first time someone forgets it."""
    import threading

    ledger = BudgetLedger(tmp_path / "concurrent.sqlite")
    errors: list[str] = []

    def _hammer():
        try:
            for _ in range(30):
                ledger.record("p", tokens=1)
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert ledger._conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 240


# --- the probe list ---------------------------------------------------------


def test_well_known_roles_are_advertised_even_when_unconfigured(tmp_path, monkeypatch):
    """A workspace that never names `codegen` still routes it, but a client
    probing `/v1/models` first may refuse an id it has not seen."""
    monkeypatch.setenv("TEST_KEY", "k")
    bare = RoutingConfig(plan="free", providers=[], roles={})
    gw = LLMGateway(routing=bare, ledger=BudgetLedger(tmp_path / "b.sqlite"), cache=None)
    ids = {m["id"] for m in handle_models(gw)["data"]}
    assert {"labpilot/codegen", "labpilot/default", "labpilot/reasoning"} <= ids
