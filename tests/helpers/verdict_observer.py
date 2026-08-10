"""What a capability actually decided, recorded while it decided it.

M20's criterion is behavioural — *this gate can say no* — and the first
mechanism answered it syntactically, by parsing the capability sources for
`passed=` calls and resolving each to a name. Seven review rounds landed inside
that parser: literal strings hidden behind `+` and `if/else`, `no_verification`
stamped from a nested block, one file holding two capabilities, a dict keyed by
name dropping a duplicate. Each fix was correct and each left the next shape
unhandled, because the parser was reconstructing from source text something the
runtime already knows exactly.

So observe it instead. Every capability reports through one type,
`TaskEvidence`, and that type carries the three facts the criterion needs:
which capability, which checks, and whether it passed. Recording them while the
suite runs turns `@pytest.mark.rejects(...)` from a claim a test author makes
into one the run has to earn — a marked test that never actually saw the gate
reject anything fails, whatever its body says.

The limit, stated rather than left to be discovered: this sees the verdicts the
suite *reaches*. A verdict no test ever exercises produces no observation and is
invisible here. That is a smaller blind spot than the parser's, and unlike the
parser's it does not masquerade as coverage — but it is real, and closing it is
tracked in `docs/research-os/autonomy-roadmap/15-gates-must-fail.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class Verdict:
    """One `TaskEvidence`, as it was constructed."""

    capability: str
    checks: tuple[str, ...]
    passed: bool


def satisfies(verdict: Verdict, claim: str) -> bool:
    """Does `verdict` make good on `@pytest.mark.rejects(claim)`?

    A claim is either a capability (`"submission"`) or a capability and one of
    its checks (`"reporting:reflect"`). The bare form is deliberately weaker
    rather than absent: some verdicts carry no check label at all, and a claim
    that could not be written for them would push those gates back out of scope.
    """
    capability, _, check = claim.partition(":")
    if verdict.passed or verdict.capability != capability:
        return False
    return not check or check in verdict.checks


def unearned(claims: Iterable[str], observed: Iterable[Verdict]) -> list[str]:
    """The claims no observation backs, in the order they were made."""
    seen = list(observed)
    return [claim for claim in claims if not any(satisfies(v, claim) for v in seen)]


@contextmanager
def recording() -> Iterator[list[Verdict]]:
    """Record every verdict reached while the block runs.

    Patches the constructor rather than a call site: capabilities build evidence
    through several helpers, and the type is the one point all of them pass
    through. `model_validate` deliberately is not covered — evidence read back
    from disk is a decision being reloaded, not one being made.

    Nesting is safe and additive: an inner block restores the outer recorder,
    and calls through it, so the outer list sees the inner block's verdicts too.
    """
    from labpilot.research_engine.execution.schemas import TaskEvidence

    observed: list[Verdict] = []
    original = TaskEvidence.__init__

    def observe(self, **fields) -> None:
        original(self, **fields)
        observed.append(Verdict(self.capability, tuple(self.checks), self.passed))

    TaskEvidence.__init__ = observe
    try:
        yield observed
    finally:
        TaskEvidence.__init__ = original


#: The live list the running test is recording into. Stashed at setup rather
#: than returned at teardown so the verdict check can run while the call-phase
#: report is still being made — an unearned marker is that test failing, not a
#: teardown error filed next to a test the summary still counts as passed.
_OBSERVED = pytest.StashKey[list]()


@pytest.fixture(autouse=True)
def verdict_observer(request):
    """Record this test's verdicts and expose them for its `rejects` markers.

    Autouse because the claim is made by a decorator, not by requesting a
    fixture: a marker whose enforcement had to be opted into would be exactly
    the unchecked claim this replaces.
    """
    with recording() as observed:
        request.node.stash[_OBSERVED] = observed
        yield observed


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Hold a passing test's `rejects` markers to what the run actually saw."""
    report = yield
    if report.when != "call" or not report.passed:
        return report

    claims = [c for mark in item.iter_markers("rejects") for c in mark.args]
    missing = unearned(claims, item.stash.get(_OBSERVED, []))
    if missing:
        observed = sorted({(v.capability, v.checks, v.passed) for v in item.stash[_OBSERVED]})
        report.outcome = "failed"
        report.longrepr = (
            f"unearned rejection marker(s): {missing}\n\n"
            "The marker claims this test proves a gate can say no, so the run has "
            "to show that gate reporting `passed=False`. It did not.\n"
            f"Verdicts observed while this test ran: {observed or 'none'}"
        )
    return report
