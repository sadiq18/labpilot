"""Tiered routing: entitlement, data policy, budget, and exhaustion behaviour."""

from __future__ import annotations

import pytest

from labpilot.llm.budget import BudgetLedger
from labpilot.llm.catalog import (
    ProviderSpec,
    RoleSpec,
    RoutingConfig,
    allowed_tiers,
    eligible_providers,
)
from labpilot.llm.router import select_route


@pytest.fixture
def ledger(tmp_path):
    with BudgetLedger(tmp_path / "budget.sqlite") as led:
        yield led


def _provider(name, **kw):
    defaults = dict(
        kind="openai_compat",
        api_key_env=f"{name.upper()}_KEY",
        models={"default": f"{name}-model"},
    )
    defaults.update(kw)
    return ProviderSpec(name=name, **defaults)


def _routing(providers, plan="free", allow_training=True, roles=None):
    return RoutingConfig(
        plan=plan,
        allow_training_on_inputs=allow_training,
        providers=providers,
        roles=roles or {},
    )


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    for name in ("FREEA", "FREEB", "PAID", "WEAK"):
        monkeypatch.setenv(f"{name}_KEY", "x")


# --- entitlement -------------------------------------------------------------


def test_free_plan_cannot_reach_paid_providers(ledger):
    routing = _routing(
        [_provider("paid", tier="paid", strong=True), _provider("freea", tier="free", strong=True)],
        plan="free",
    )
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider is not None
    assert decision.provider.name == "freea"


def test_pro_plan_prefers_paid_over_free(ledger):
    routing = _routing(
        [_provider("freea", tier="free", strong=True), _provider("paid", tier="paid", strong=True)],
        plan="pro",
    )
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider.name == "paid"


def test_enterprise_excludes_free_tiers_entirely():
    """Free tiers may train on inputs; enterprise data cannot go there."""
    assert "free" not in allowed_tiers("enterprise")
    assert "paid" in allowed_tiers("enterprise")


def test_unknown_plan_gets_most_restrictive_tiers():
    assert allowed_tiers("platinum-deluxe") == allowed_tiers("free")


# --- data policy -------------------------------------------------------------


def test_training_on_inputs_can_be_refused_regardless_of_plan(ledger):
    routing = _routing(
        [
            _provider("freea", tier="free", strong=True, trains_on_input=True),
            _provider("freeb", tier="free", strong=True, trains_on_input=False),
        ],
        plan="free",
        allow_training=False,
    )
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider.name == "freeb"


def test_provider_without_credentials_is_not_eligible(monkeypatch):
    monkeypatch.delenv("FREEA_KEY", raising=False)
    routing = _routing([_provider("freea", tier="free", strong=True)])
    assert eligible_providers(routing, "reasoning") == []


def test_local_provider_needs_no_credentials():
    routing = _routing([_provider("ollama", tier="local", api_key_env="")])
    assert [p.name for p in eligible_providers(routing, "summarize")] == ["ollama"]


# --- budget ------------------------------------------------------------------


def test_exhausted_rpm_moves_to_next_provider(ledger):
    routing = _routing(
        [
            _provider("freea", tier="free", strong=True, rpm=2),
            _provider("freeb", tier="free", strong=True, rpm=10),
        ]
    )
    for _ in range(2):
        ledger.record("freea")
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider.name == "freeb"


def test_all_exhausted_reports_wait_not_failure(ledger):
    routing = _routing(
        [_provider("freea", tier="free", strong=True, rpm=1)],
        roles={"reasoning": RoleSpec(requires_strong=True, on_exhaustion="wait")},
    )
    ledger.record("freea")
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider is None
    assert 0 < decision.wait_seconds <= 60
    assert "rate limit" in decision.reason


def test_cooldown_blocks_a_provider(ledger):
    routing = _routing([_provider("freea", tier="free", strong=True, rpm=100)])
    ledger.cool_down("freea", 30.0, reason="429")
    decision = select_route(routing, "reasoning", ledger)
    assert decision.provider is None
    assert "cooling down" in decision.reason


def test_daily_limit_is_tracked_separately_from_rpm(ledger):
    routing = _routing([_provider("freea", tier="free", strong=True, rpm=100, rpd=3)])
    now = 1_000_000.0
    for i in range(3):
        ledger.record("freea", now=now - 3600 * (i + 1))  # spread out, so rpm is clear
    decision = select_route(routing, "reasoning", ledger, now=now)
    assert decision.provider is None
    assert "daily limit" in decision.reason


# --- exhaustion behaviour ----------------------------------------------------


def test_reasoning_waits_rather_than_using_a_weak_model(ledger):
    """A weak model writing an experiment records a false negative: the system
    concludes the *technique* failed when really the writer did."""
    routing = _routing(
        [
            _provider("freea", tier="free", strong=True, rpm=1),
            _provider("weak", tier="free", strong=False, rpm=100),
        ],
        roles={"codegen": RoleSpec(requires_strong=True, on_exhaustion="wait")},
    )
    ledger.record("freea")
    decision = select_route(routing, "codegen", ledger)
    assert decision.provider is None, "must not silently fall back to the weak model"
    assert decision.wait_seconds > 0


def test_summarize_degrades_to_a_weak_model(ledger):
    routing = _routing(
        [
            _provider("freea", tier="free", strong=True, rpm=1),
            _provider("weak", tier="free", strong=False, rpm=100),
        ],
        roles={"summarize": RoleSpec(requires_strong=True, on_exhaustion="degrade")},
    )
    ledger.record("freea")
    decision = select_route(routing, "summarize", ledger)
    assert decision.provider.name == "weak"
    assert decision.degraded is True


def test_degradation_is_always_reported(ledger):
    """Provenance: an experiment served by a downgraded model must be traceable."""
    routing = _routing(
        [
            _provider("freea", tier="free", strong=True, rpm=1),
            _provider("weak", tier="free", strong=False, rpm=100),
        ],
        roles={"summarize": RoleSpec(requires_strong=True, on_exhaustion="degrade")},
    )
    ledger.record("freea")
    decision = select_route(routing, "summarize", ledger)
    assert decision.degraded is True
    assert decision.reason


def test_budget_state_survives_a_restart(tmp_path):
    """Campaigns span processes; a fresh run must not forget today's spend."""
    path = tmp_path / "budget.sqlite"
    with BudgetLedger(path) as first:
        first.record("freea")
        first.record("freea")
    with BudgetLedger(path) as second:
        assert second.availability("freea", rpm=2).ok is False
