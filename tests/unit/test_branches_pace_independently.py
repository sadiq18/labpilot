"""K branches hitting one provider's rate limit wait, they do not fail (M11).

Design §5 closed the budget question by declining to build anything: no
per-branch pre-split, no `fitroute` change. The argument was that
`RoleBoundClient.complete(allow_wait=True)` already sleeps `wait_seconds` and
retries instead of raising, that `BudgetLedger` is `RLock`-guarded, and that
each branch runs on its own OS thread — so a branch that hits a rate limit
sleeps on its own thread while its siblings keep working.

Every clause of that is a property of code this milestone does not own, which
is why it needs a guard rather than a note. `allow_wait` ceasing to be the
default, a lock dropped from the ledger, or the gateway sleeping while holding
one, would each turn a paced fan-out into a failing one and nothing in M11
would notice.

Test-only by design: §5's resolution was "no production code", and adding some
here would quietly reverse a decision rather than verify it.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from fitroute.adapters import Completion
from fitroute.budget import BudgetLedger
from fitroute.catalog import ProviderSpec, RoleSpec, RoutingConfig
from fitroute.gateway import LLMGateway, RoleUnavailable

#: `rpm` windows are a real minute, so a real wait would cost the suite 60s a
#: branch. The clock is driven instead — see `_FakeClock`.
_WINDOW_S = 60.0

_BRANCHES = 3


class _FakeClock:
    """A clock the gateway's sleeps advance, shared by every branch.

    Lets a paced retry land in the next rate-limit window without the suite
    waiting for one, while leaving the concurrency real: the branches are
    genuine threads and this is the state they contend over, so it is guarded
    the way the ledger it stands in for is.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self.overlap_peak = 0
        self._sleeping = 0
        self._guard = threading.Lock()

    def time(self) -> float:
        with self._guard:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._guard:
            self.sleeps.append(seconds)
            self._sleeping += 1
            self.overlap_peak = max(self.overlap_peak, self._sleeping)
            self.now += seconds
        # A real yield, so waiting branches genuinely coexist rather than each
        # running start-to-finish while the others are descheduled.
        time.sleep(0.005)
        with self._guard:
            self._sleeping -= 1


class _CountingAdapter:
    """The one provider every branch shares."""

    supports_json_mode = True

    def __init__(self) -> None:
        self.calls = 0
        self._guard = threading.Lock()

    def complete(self, system, user, *, model, temperature, json_mode=False):
        del system, user, model, temperature, json_mode
        with self._guard:
            self.calls += 1
        return Completion("{}", prompt_tokens=1, completion_tokens=1)


@pytest.fixture
def ledger(tmp_path):
    with BudgetLedger(tmp_path / "budget.sqlite") as led:
        yield led


def _routing(*, rpm: int = 1, max_wait_seconds: float = _WINDOW_S * 10) -> RoutingConfig:
    """One provider with room for one call a minute, and nowhere to fail over.

    A second provider would let the branches degrade sideways instead of
    waiting, and the waiting is the property under test. `on_exhaustion="wait"`
    for the same reason — the default is `degrade`, which reports no wait when
    there is nothing to degrade to.
    """
    return RoutingConfig(
        plan="free",
        providers=[
            ProviderSpec(
                name="only",
                tier="local",
                strong=True,
                caps={"structured_output"},
                rpm=rpm,
                models={"default": "m-only"},
            )
        ],
        roles={
            "default": RoleSpec(
                requires={"structured_output"},
                on_exhaustion="wait",
                max_wait_seconds=max_wait_seconds,
            )
        },
    )


@pytest.fixture
def clock(monkeypatch):
    """Drive `fitroute`'s two clocks: the ledger reads one, the gateway sleeps
    on the other, and a sleep has to move the ledger's or the retry lands in
    the same exhausted window it just left."""
    fake = _FakeClock()
    monkeypatch.setattr("fitroute.budget.time", SimpleNamespace(time=fake.time))
    monkeypatch.setattr(
        "fitroute.gateway.time",
        SimpleNamespace(sleep=fake.sleep, monotonic=time.monotonic),
    )
    return fake


def _gateway(monkeypatch, ledger, *, routing=None):
    adapter = _CountingAdapter()
    monkeypatch.setattr("fitroute.gateway.build_adapter", lambda *a, **k: adapter)
    gateway = LLMGateway(
        routing or _routing(), ledger, credential_resolver=lambda name: "key"
    )
    return gateway, adapter


def _run_branches(gateway, count: int) -> tuple[list[str], list[BaseException]]:
    """Call `complete` from `count` threads released together.

    Threads, not coroutines: §5's argument rests on each branch owning an OS
    thread, which is what `anyio.to_thread.run_sync` gives the real fan-out. On
    one thread the gateway's blocking sleep would serialise the branches, and
    this would pass while proving nothing about a fan-out.
    """
    results: list[str] = []
    errors: list[BaseException] = []
    guard = threading.Lock()
    start = threading.Barrier(count)

    def one() -> None:
        client = gateway.for_role("default")
        start.wait()
        try:
            answer = client.complete("s", "u", json_mode=True)
            with guard:
                results.append(answer)
        except BaseException as exc:  # noqa: BLE001 — the failures are the result
            with guard:
                errors.append(exc)

    threads = [threading.Thread(target=one) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


def test_two_branches_on_a_one_call_window_are_both_served(
    monkeypatch, ledger, clock
) -> None:
    """Pacing, working: one branch takes the slot, the other waits out the
    window and is served. Were `allow_wait` not the default, the second would
    die on a rate limit rather than on its science, and the fan-out would
    record it as a failed experiment.
    """
    gateway, adapter = _gateway(monkeypatch, ledger)

    results, errors = _run_branches(gateway, 2)

    assert errors == [], f"a branch failed instead of waiting: {errors}"
    assert len(results) == 2
    assert adapter.calls == 2
    assert len(clock.sleeps) == 1, "the second branch did not wait for its turn"
    assert 0 < clock.sleeps[0] <= _WINDOW_S


def test_waiting_branches_wait_at_the_same_time(monkeypatch, ledger, clock) -> None:
    """What makes pacing tolerable: a branch sleeping on a rate limit holds
    nothing its siblings need, so the waits overlap. If the gateway slept
    while holding the ledger — or the ledger's lock were taken across the
    sleep — the waits would queue and a K-branch fan-out would pay K windows
    instead of one.
    """
    gateway, _ = _gateway(monkeypatch, ledger)

    _run_branches(gateway, _BRANCHES)

    assert clock.overlap_peak > 1, (
        "no two branches were ever waiting at once — they are queueing behind "
        "a held lock, not pacing independently"
    )


@pytest.mark.parametrize(
    ("branches", "rpm", "served"),
    [
        (2, 1, 2),  # one immediately, one after a wait
        (3, 1, 2),  # the third has no second wait left
        (4, 1, 2),
        (3, 2, 3),
        (4, 2, 4),
        (5, 2, 4),
    ],
)
def test_a_fan_out_wider_than_two_windows_loses_branches(
    monkeypatch, ledger, clock, branches: int, rpm: int, served: int
) -> None:
    """The measured limit of §5's resolution: **served = min(K, 2 x rpm)**.

    A branch gets exactly *one* wait. `_complete_once` sleeps once, re-selects,
    and raises `RoleUnavailable` if the provider is still spent; the outer
    `complete` retry loop passes `allow_wait=False` on every attempt after the
    first, so nothing waits twice. One window's worth of branches is served
    immediately and one more window's worth after that single wait — the rest
    fail.

    §5 called this "a small thundering-herd retry burst ... not perfectly
    efficient", which reads as a throughput note. It is not: the excess
    branches do not run slower, they fail, and the fan-out records each as a
    failed experiment against the circuit breaker. `--branches 4` on a
    one-call-per-minute provider spends four worktrees to get two results and
    two breaker strikes.

    Pinned rather than fixed: fixing it means retry-with-wait in `fitroute`,
    which §5 explicitly scoped out. Pinned so that choosing K against a known
    rpm is a decision someone can look up instead of rediscover.
    """
    gateway, adapter = _gateway(monkeypatch, ledger, routing=_routing(rpm=rpm))

    results, errors = _run_branches(gateway, branches)

    assert len(results) == served
    assert adapter.calls == served
    assert len(errors) == branches - served
    assert all(isinstance(e, RoleUnavailable) for e in errors), errors


def test_a_wait_longer_than_the_bound_fails_instead_of_hanging(
    monkeypatch, ledger, clock
) -> None:
    """The other half of the contract, and why relying on waiting is safe: a
    branch never sleeps indefinitely on an exhausted provider. It raises, and
    the fan-out records one failed branch rather than hanging the campaign —
    `max_wait_seconds` exists because an unbounded wait in an unattended run is
    indistinguishable from a hang.
    """
    gateway, adapter = _gateway(
        monkeypatch, ledger, routing=_routing(max_wait_seconds=1.0)
    )
    client = gateway.for_role("default")

    client.complete("s", "u", json_mode=True)  # spends the only slot

    with pytest.raises(RoleUnavailable):
        client.complete("s", "u", json_mode=True)
    assert adapter.calls == 1
    assert clock.sleeps == [], "it slept despite the wait exceeding the bound"
