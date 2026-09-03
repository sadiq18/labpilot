"""A background caller must run out of budget before the foreground one (M16).

`BudgetLedger.availability` answered identically for every caller, so nothing
implemented M16's third trap: "producer work should be the *lower* priority
claim on the ledger". A sweep is minutes of `reasoning` calls, and finishing
one at the cost of the campaign's next plan is a bad trade in both directions.
"""

from __future__ import annotations

import pytest

from fitroute.budget import BudgetLedger


@pytest.fixture
def ledger(tmp_path):
    with BudgetLedger(tmp_path / "budget.sqlite") as led:
        yield led


def _spend(ledger: BudgetLedger, calls: int, *, provider: str = "free-tier") -> None:
    for _ in range(calls):
        ledger.record(provider, tokens=100)


def test_a_reserve_refuses_before_the_real_limit_does(ledger) -> None:
    _spend(ledger, 8)  # of 10/min

    assert ledger.availability("free-tier", rpm=10).ok is True
    assert ledger.availability("free-tier", rpm=10, reserve=0.2).ok is False


def test_the_foreground_caller_still_sees_the_real_limit(ledger) -> None:
    """The whole point: one caller yields, the other does not."""
    _spend(ledger, 8)

    background = ledger.availability("free-tier", rpm=10, reserve=0.2)
    foreground = ledger.availability("free-tier", rpm=10)

    assert background.ok is False
    assert foreground.ok is True


def test_the_reserve_applies_to_whichever_window_binds(ledger) -> None:
    """Not a call count: a token-metered provider binds on `tpm`, where
    "hold back five calls" would mean nothing.
    """
    _spend(ledger, 3)  # 300 tokens, well under a 1000/min budget

    assert ledger.availability("free-tier", tpm=1000).ok is True
    assert ledger.availability("free-tier", tpm=1000, reserve=0.75).ok is False
    assert ledger.availability("free-tier", rpd=4, reserve=0.5).ok is False


def test_a_reserve_never_takes_the_provider_away_entirely(ledger) -> None:
    """It exists to make a background caller yield sooner, not to lock it out
    of an idle provider — `int(limit * 0.01)` would floor to zero.
    """
    assert ledger.availability("free-tier", rpm=10, reserve=0.99).ok is True


def test_no_reserve_is_the_behaviour_every_existing_caller_had(ledger) -> None:
    _spend(ledger, 9)

    assert ledger.availability("free-tier", rpm=10).ok is True
    assert ledger.availability("free-tier", rpm=10, reserve=0.0).ok is True

    _spend(ledger, 1)
    assert ledger.availability("free-tier", rpm=10).ok is False


def test_a_reserved_refusal_quotes_the_real_limit_not_the_reserved_one(ledger) -> None:
    """The reason string reaches `RouteDecision.reason`, the producer's skip
    reason and `research doctor`. Naming the reduced number there reads as the
    provider's published quota and cannot be reconciled with its dashboard.
    """
    _spend(ledger, 8)

    refused = ledger.availability("free-tier", rpm=10, reserve=0.2)

    assert refused.ok is False
    assert "rate limit 10/min" in refused.reason
    assert "8" not in refused.reason.split("holding")[0]
    assert "holding 20% in reserve" in refused.reason


def test_an_unreserved_refusal_says_nothing_about_a_reserve(ledger) -> None:
    _spend(ledger, 10)

    refused = ledger.availability("free-tier", rpm=10)

    assert refused.ok is False
    assert "reserve" not in refused.reason
