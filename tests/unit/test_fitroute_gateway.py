"""fitroute v0.1 — capability preflight, credentials, and the gateway.

Every test here maps to a failure measured in a real rogii campaign, not to an
imagined one. The comment above each says which.
"""

from __future__ import annotations

import json

import pytest

from fitroute.adapters import Completion, build_adapter
from fitroute.budget import BudgetLedger
from fitroute.cache import PromptCache
from fitroute.catalog import ProviderSpec, RoleSpec, RoutingConfig, eligible_providers
from fitroute.gateway import LLMGateway, RoleUnavailable
from fitroute.select import select_route


@pytest.fixture
def ledger(tmp_path):
    with BudgetLedger(tmp_path / "budget.sqlite") as led:
        yield led


class _FakeAdapter:
    """Records what the gateway asked for."""

    supports_json_mode = True

    def __init__(self, text='{"ok": true}', tokens=(11, 7)):
        self.text = text
        self.tokens = tokens
        self.calls: list[dict] = []

    def complete(self, system, user, *, model, temperature, json_mode=False):
        self.calls.append(
            {"system": system, "user": user, "model": model, "json_mode": json_mode}
        )
        return Completion(self.text, prompt_tokens=self.tokens[0], completion_tokens=self.tokens[1])


def _gateway(monkeypatch, providers, roles=None, *, cache=None, ledger=None, adapter=None):
    routing = RoutingConfig(plan="free", providers=providers, roles=roles or {})
    adapter = adapter or _FakeAdapter()
    monkeypatch.setattr("fitroute.gateway.build_adapter", lambda *a, **k: adapter)
    gw = LLMGateway(routing, ledger, cache=cache, credential_resolver=lambda name: "key")
    return gw, adapter


def _ollama(**kw):
    defaults = dict(
        name="ollama",
        kind="ollama",
        tier="local",
        caps={"structured_output"},
        models={"default": "qwen2.5-coder:14b"},
    )
    defaults.update(kw)
    return ProviderSpec(**defaults)


# --- capability preflight ----------------------------------------------------


def test_model_without_structured_output_is_not_a_candidate(ledger):
    """The measured failure: a model answered a JSON-only prompt in English
    prose, returned HTTP 200, and the reply was discarded — 3 of 3 campaigns."""
    routing = RoutingConfig(
        plan="free",
        providers=[_ollama(caps=set())],
        roles={"codegen": RoleSpec(requires={"structured_output"})},
    )
    assert eligible_providers(routing, "codegen") == []

    decision = select_route(routing, "codegen", ledger)
    assert decision.provider is None
    assert "structured_output" in decision.reason, "must name the capability that rejected it"


def _degrade_routing(*providers):
    return RoutingConfig(
        plan="free",
        providers=list(providers),
        roles={
            "summarize": RoleSpec(
                requires_strong=True, requires={"structured_output"}, on_exhaustion="degrade"
            )
        },
    )


def test_degrading_relaxes_strength(ledger):
    """The control for the test below: degrading works when the weak provider
    is capable. Without this, that test could pass for any reason at all."""
    routing = _degrade_routing(
        _ollama(name="strong", strong=True, rpm=1),
        _ollama(name="weak_but_capable"),
    )
    ledger.record("strong")
    decision = select_route(routing, "summarize", ledger)
    assert decision.provider.name == "weak_but_capable"
    assert decision.degraded is True


def test_capability_is_never_relaxed_by_degrading(ledger):
    """Strength may be relaxed on exhaustion; capability may not — degrading to
    a model that cannot do the job is not a degraded result.

    Both providers are `tier="local"` so credentials cannot be the reason the
    candidate list is empty. An earlier version used `tier="free"` with no
    api_key_env, which made *both* ineligible for lack of credentials — so a
    regression that degraded onto the incapable model would still have passed.
    """
    routing = _degrade_routing(
        _ollama(name="strong", strong=True, rpm=1),
        _ollama(name="weak_no_json", caps=set()),
    )
    ledger.record("strong")
    decision = select_route(routing, "summarize", ledger)

    assert decision.provider is None, "must not degrade onto a model lacking the capability"
    # The reason is the strong provider's rate limit, not a capability message:
    # a capable provider does exist, it is merely exhausted, so waiting for it
    # is the correct outcome.
    assert decision.wait_seconds > 0


def test_no_candidate_reason_names_the_filter(ledger):
    """'No eligible provider' with no cause is what sends people to read the
    router's source."""
    routing = RoutingConfig(
        plan="free",
        providers=[ProviderSpec(name="groq", api_key_env="GROQ_API_KEY", models={"default": "m"})],
    )
    decision = select_route(routing, "reasoning", ledger)
    assert "GROQ_API_KEY" in decision.reason


# --- credentials -------------------------------------------------------------


def test_dotenv_only_key_is_visible_via_resolver(monkeypatch):
    """pydantic-settings loads .env into a Settings object and never exports to
    os.environ, so an env-only catalog cannot see a key the user just added."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    routing = RoutingConfig(
        plan="free",
        providers=[
            ProviderSpec(
                name="groq",
                api_key_env="GROQ_API_KEY",
                tier="free",
                caps={"structured_output"},
                models={"default": "m"},
            )
        ],
        roles={"reasoning": RoleSpec(requires={"structured_output"})},
    )
    assert eligible_providers(routing, "reasoning") == []

    found = eligible_providers(
        routing, "reasoning", credential_resolver=lambda name: "sk-test-from-dotenv"
    )
    assert [p.name for p in found] == ["groq"]


# --- the gateway -------------------------------------------------------------


def test_json_mode_reaches_the_adapter(monkeypatch, ledger):
    """The obvious wiring drops this flag, and dropping it is what produced the
    3-of-3 fallback rate."""
    gw, adapter = _gateway(monkeypatch, [_ollama()], ledger=ledger)
    gw.for_role("default").complete("sys", "user", json_mode=True)
    assert adapter.calls[0]["json_mode"] is True


def test_every_call_is_metered(monkeypatch, ledger):
    """Before this, BudgetLedger.record had no caller anywhere in the system."""
    gw, _ = _gateway(monkeypatch, [_ollama()], ledger=ledger)
    client = gw.for_role("default")
    for _ in range(3):
        client.complete("sys", "user")
    assert ledger.availability("ollama", rpm=3).ok is False, "3 calls should exhaust rpm=3"


def test_failed_call_is_still_metered(monkeypatch, ledger):
    """A rejected call consumed quota on most providers."""

    class _Boom:
        supports_json_mode = True

        def complete(self, *a, **k):
            raise RuntimeError("HTTP 500")

    gw, _ = _gateway(monkeypatch, [_ollama()], ledger=ledger, adapter=_Boom())
    with pytest.raises(RuntimeError):
        gw.for_role("default").complete("sys", "user")
    assert ledger.availability("ollama", rpm=1).ok is False


def test_cache_hit_does_not_spend_quota(monkeypatch, tmp_path, ledger):
    """The measured state before this: one cache row across nine campaigns,
    because caching was the caller's job and callers didn't."""
    cache = PromptCache(tmp_path / "llm.sqlite")
    gw, adapter = _gateway(monkeypatch, [_ollama()], cache=cache, ledger=ledger)
    client = gw.for_role("default")

    first = client.complete("sys", "user")
    second = client.complete("sys", "user")

    assert first == second
    assert len(adapter.calls) == 1, "second call must be served from cache"
    assert client.last_served.cache_hit is True
    assert ledger.availability("ollama", rpm=2).ok is True, "cache hit must not spend quota"


def test_served_stamp_identifies_the_model(monkeypatch, ledger):
    """Without this a failed hypothesis cannot be attributed to the idea rather
    than the writer."""
    gw, _ = _gateway(monkeypatch, [_ollama()], ledger=ledger)
    client = gw.for_role("codegen")
    client.complete("sys", "user")

    served = client.last_served
    assert served.provider == "ollama"
    assert served.model == "qwen2.5-coder:14b"
    assert served.role == "codegen"
    assert served.tokens == 18


def test_unmetered_response_is_not_reported_as_zero(monkeypatch, ledger):
    """Unknown is not zero: zero-filling is how a budget cap gets blown while
    the dashboard looks calm."""
    unmetered = _FakeAdapter(tokens=(None, None))
    gw, _ = _gateway(monkeypatch, [_ollama()], ledger=ledger, adapter=unmetered)
    client = gw.for_role("default")
    client.complete("sys", "user")
    assert client.last_served.tokens is None


def test_unavailable_role_raises_rather_than_hanging(monkeypatch, ledger):
    """An unbounded wait in an unattended campaign is indistinguishable from a
    hang, and the operator can only kill a run whose state they cannot see."""
    gw, _ = _gateway(
        monkeypatch,
        [_ollama(rpd=1)],
        roles={"codegen": RoleSpec(on_exhaustion="wait", max_wait_seconds=5.0)},
        ledger=ledger,
    )
    client = gw.for_role("codegen")
    client.complete("sys", "user")

    with pytest.raises(RoleUnavailable) as exc:
        client.complete("sys", "user")
    assert "codegen" in str(exc.value)
    assert "daily limit" in str(exc.value)


# --- adapters ----------------------------------------------------------------


def test_fail_raises_immediately_instead_of_waiting(ledger):
    """`fail` reports no wait, so a caller pacing on wait_seconds raises now.

    Otherwise `fail` sleeps up to max_wait_seconds first and is indistinguishable
    from `wait`.
    """
    routing = RoutingConfig(
        plan="free",
        providers=[_ollama(rpm=1)],
        roles={"strict": RoleSpec(on_exhaustion="fail", max_wait_seconds=900.0)},
    )
    ledger.record("ollama")
    decision = select_route(routing, "strict", ledger)

    assert decision.provider is None
    assert decision.wait_seconds == 0.0, "fail must not ask the caller to wait"

    waiting = RoutingConfig(
        plan="free",
        providers=[_ollama(rpm=1)],
        roles={"patient": RoleSpec(on_exhaustion="wait")},
    )
    assert select_route(waiting, "patient", ledger).wait_seconds > 0


def test_build_adapter_rejects_unknown_kind():
    with pytest.raises(ValueError, match="openai_compat"):
        build_adapter("magic", base_url="http://x")


def test_completion_distinguishes_unmetered_from_zero():
    assert Completion("x").metered is False
    assert Completion("x", prompt_tokens=0, completion_tokens=0).metered is True


# --- structured output via tool calling --------------------------------------
#
# Several of the strongest free models on OpenRouter advertise `tools` but not
# `response_format` (nvidia/nemotron-3-ultra-550b-a55b:free, cohere/north-mini-
# code:free, poolside/laguna-{s,xs}-2.1:free, checked 2026-08-07). With
# structured_output a hard precondition, a response_format-only adapter rejects
# exactly the models best suited to codegen.


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch, body: dict) -> list[dict]:
    """Monkeypatch the transport; return the list request payloads land in."""
    sent: list[dict] = []

    def _fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(body)

    monkeypatch.setattr("fitroute.adapters.urllib.request.urlopen", _fake_urlopen)
    return sent


def _tool_call_body(arguments):
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"function": {"name": "emit_json", "arguments": arguments}}],
                }
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }


def _envelope(payload: str):
    """What a provider sends back: the payload inside the schema's one property."""
    return _tool_call_body(json.dumps({"json": payload}))


def test_tools_route_forces_a_single_function_call(monkeypatch):
    """`tools` alone only *offers* the function. Without tool_choice the model
    is free to answer in prose, which is the failure the cap exists to prevent."""
    sent = _capture(monkeypatch, _envelope('{"ok": true}'))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    adapter.complete("sys", "user", model="m", temperature=0.1, json_mode=True)

    payload = sent[0]
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "emit_json"}}
    assert [t["function"]["name"] for t in payload["tools"]] == ["emit_json"]
    assert "response_format" not in payload, (
        "sending both to a model that rejects response_format fails the whole call"
    )


def test_forced_tool_declares_a_property_to_carry_the_payload(monkeypatch):
    """Measured 2026-08-07 on gemini-3.5-flash-lite: an open object schema
    (`additionalProperties: true`, no `properties`) yields a forced call whose
    arguments are `{}`. Providers validate arguments against the declared schema
    and drop what it does not name — an empty answer that parses, which is worse
    than a failure. The payload needs a declared property to travel in."""
    sent = _capture(monkeypatch, _envelope("{}"))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)

    schema = sent[0]["tools"][0]["function"]["parameters"]
    assert schema["required"] == ["json"]
    assert schema["properties"]["json"]["type"] == "string"


def test_tools_route_returns_the_payload_not_the_envelope(monkeypatch):
    """Callers json.loads the result, so they must get what the prompt asked
    for — not `{"json": "..."}` wrapped around it."""
    _capture(monkeypatch, _envelope('{"technique": "target-encoding"}'))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    result = adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert json.loads(result.text) == {"technique": "target-encoding"}
    assert result.total_tokens == 17, "usage is read the same way on both routes"


def test_tools_route_survives_a_payload_containing_code(monkeypatch):
    """Codegen is the motivating role: the payload holds source with newlines and
    quotes, which has to come back through the envelope byte-identical."""
    source = 'def add(a, b):\n    """Add."""\n    return a + b\n'
    _capture(monkeypatch, _envelope(json.dumps({"code": source})))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    result = adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert json.loads(result.text)["code"] == source


def test_tools_route_accepts_object_valued_arguments(monkeypatch):
    """The OpenAI schema says `arguments` is a string; several compatible
    endpoints send the object. Both mean the same thing."""
    _capture(monkeypatch, _tool_call_body({"json": '{"ok": true}'}))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    result = adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert json.loads(result.text) == {"ok": True}


def test_tools_route_accepts_a_payload_nested_instead_of_serialised(monkeypatch):
    """A model that ignores the envelope's `type: string` and nests the object
    has still answered the question."""
    _capture(monkeypatch, _tool_call_body(json.dumps({"json": {"ok": True}})))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    result = adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert json.loads(result.text) == {"ok": True}


def test_tools_route_raises_when_the_envelope_is_empty(monkeypatch):
    """The `{}` failure above, reaching the caller: it parses as JSON but says
    nothing, so it must be an error here rather than an empty result downstream."""
    _capture(monkeypatch, _tool_call_body("{}"))
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    with pytest.raises(RuntimeError, match="without 'json'"):
        adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)


def test_tools_route_raises_when_the_forced_call_is_ignored(monkeypatch):
    """An endpoint that answers in content despite tool_choice does not have the
    capability its catalog entry claims. Falling back to content would hide a
    wrong entry and hand prose to a caller about to parse it as JSON."""
    _capture(monkeypatch, {"choices": [{"message": {"content": "Sure! Here is the JSON:"}}]})
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    with pytest.raises(RuntimeError, match="ignored the forced tool call"):
        adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)


def test_tools_route_sends_no_tools_when_json_is_not_asked_for(monkeypatch):
    """A prose call on a tools-route provider is an ordinary call — paying for a
    tool definition (and forcing a call) would change what the model returns."""
    sent = _capture(monkeypatch, {"choices": [{"message": {"content": "hello"}}]})
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="tools")
    result = adapter.complete("s", "u", model="m", temperature=0.1)
    assert "tools" not in sent[0] and "tool_choice" not in sent[0]
    assert result.text == "hello"


def test_response_format_stays_the_default_route(monkeypatch):
    """The existing path is simpler and better tested; adding a second route
    must not quietly migrate providers that never asked for it."""
    sent = _capture(monkeypatch, {"choices": [{"message": {"content": "{}"}}]})
    adapter = build_adapter("openai_compat", base_url="https://x/v1", api_key="k")
    adapter.complete("s", "u", model="m", temperature=0.1, json_mode=True)
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert "tools" not in sent[0]


def test_provider_json_via_reaches_the_adapter(monkeypatch, ledger):
    """The wiring this feature is worth nothing without: a catalog entry saying
    `tools` must actually change the request the gateway makes."""
    seen: dict = {}

    def _spy(kind, **kwargs):
        seen.update(kwargs)
        return _FakeAdapter()

    monkeypatch.setattr("fitroute.gateway.build_adapter", _spy)
    routing = RoutingConfig(
        plan="free",
        providers=[_ollama(name="tools-provider", json_via="tools")],
    )
    gw = LLMGateway(routing, ledger, credential_resolver=lambda name: "key")
    gw.for_role("default").complete("sys", "user", json_mode=True)
    assert seen["json_via"] == "tools"


def test_unknown_json_via_is_rejected():
    """Same reasoning as an unknown name in routing.use: falling back to the
    default would send response_format to an endpoint that ignores it."""
    with pytest.raises(ValueError, match="json_via"):
        build_adapter("openai_compat", base_url="https://x/v1", api_key="k", json_via="toolz")
