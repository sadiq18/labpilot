"""Falling back is temporary: the best provider is retried on every call.

The property asked for on 2026-08-07 — "if a better model is available again,
better to start from the beginning than going down". It already holds, and
this file exists so it keeps holding.

`select_route` is stateless: it walks `eligible_providers` from the top on
*every* call and returns the first whose budget allows. There is no cursor and
no memory of where the last call landed, so the moment a preferred provider's
rate window rolls over it is used again. The alternative — a walking cursor —
would degrade permanently after one busy minute, which is the failure this
pins against.
"""

from __future__ import annotations

import pytest

from fitroute.budget import BudgetLedger
from fitroute.catalog import ProviderSpec, RoleSpec, RoutingConfig
from fitroute.select import select_route


@pytest.fixture
def ledger(tmp_path):
    with BudgetLedger(tmp_path / "budget.sqlite") as led:
        yield led


def _routing() -> RoutingConfig:
    """Best-first order: one call/min each, so exhaustion is easy to trigger."""
    return RoutingConfig(
        plan="free",
        providers=[
            ProviderSpec(
                name="best", tier="local", strong=True, caps={"structured_output"},
                rpm=1, models={"default": "m-best"},
            ),
            ProviderSpec(
                name="second", tier="local", strong=True, caps={"structured_output"},
                rpm=1, models={"default": "m-second"},
            ),
            ProviderSpec(
                name="floor", tier="local", strong=True, caps={"structured_output"},
                models={"default": "m-floor"},
            ),
        ],
        roles={"default": RoleSpec(requires={"structured_output"})},
    )


def _pick(routing, ledger, *, now):
    return select_route(routing, "default", ledger, now=now).provider.name


def test_the_first_call_takes_the_best_provider(ledger):
    assert _pick(_routing(), ledger, now=1000.0) == "best"


def test_it_walks_down_as_each_provider_is_spent(ledger):
    routing = _routing()
    order = []
    for _ in range(3):
        name = _pick(routing, ledger, now=1000.0)
        order.append(name)
        ledger.record(name, now=1000.0)
    assert order == ["best", "second", "floor"]


def test_it_returns_to_the_best_provider_once_its_window_rolls_over(ledger):
    """The asked-for property. A walking cursor would answer 'second' here."""
    routing = _routing()
    ledger.record("best", now=1000.0)

    # Within the same minute, `best` is spent: fall to `second`.
    assert _pick(routing, ledger, now=1000.5) == "second"

    # A minute later its rpm window has rolled over — start from the top again.
    assert _pick(routing, ledger, now=1070.0) == "best", (
        "routing stayed on the fallback after the preferred provider recovered"
    )


def test_recovery_survives_falling_all_the_way_to_the_floor(ledger):
    """Even after exhausting every tier, the next window starts at the best."""
    routing = _routing()
    for name in ("best", "second"):
        ledger.record(name, now=1000.0)
    assert _pick(routing, ledger, now=1000.5) == "floor"
    assert _pick(routing, ledger, now=1070.0) == "best"


def test_selection_is_stateless_across_calls(ledger):
    """Ten identical calls in a fresh window all choose the best provider —
    nothing accumulates that would nudge routing downward over time."""
    routing = _routing()
    picks = {_pick(routing, ledger, now=2000.0 + i * 120) for i in range(10)}
    assert picks == {"best"}
