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

#: How much of that list belongs to setup. Everything after it is the test body.
_CALL_START = pytest.StashKey[int]()


def _summarise(verdicts: list[Verdict]) -> str:
    return str(sorted({(v.capability, v.checks, v.passed) for v in verdicts})) or "none"


def _fail(report, message: str) -> None:
    report.outcome = "failed"
    report.longrepr = message


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
def pytest_runtest_call(item):
    """Mark where the test body starts within this test's recording.

    The recorder is autouse, so it is installed before the test's other fixtures
    and sees the verdicts they produce during setup. Those must not earn the
    marker — the claim is that *this test* proves a gate can say no, and a
    fixture saying it does not make that true. Reported reviewing PR #121,
    round 8; verified latent at the time rather than live.

    Recorded as an index rather than by narrowing what is recorded, because a
    capability driven through a fixture still has to be observable to the
    diagnostic below when the claim goes unmet.
    """
    item.stash[_CALL_START] = len(item.stash.get(_OBSERVED, []))
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Hold a passing test's `rejects` markers to what its body actually caused."""
    report = yield
    if report.when != "call" or not report.passed:
        return report

    marks = list(item.iter_markers("rejects"))
    if not marks:
        return report

    claims = [c for mark in marks for c in mark.args]
    # A marker carrying nothing to check used to pass, which made writing it
    # wrong quieter than not writing it at all. Reported reviewing PR #121,
    # round 8.
    if not claims or any(not str(claim).strip() for claim in claims):
        _fail(
            report,
            "a `rejects` marker with no argument names no gate, so nothing can be "
            "checked and the claim is untestable.\n"
            'Write `@pytest.mark.rejects("<capability>")` or '
            '`@pytest.mark.rejects("<capability>:<check>")`.',
        )
        return report

    # Every read is `.get`. The mismatched pair here — one `.get`, one subscript,
    # two lines apart — raised `KeyError` inside this wrapper whenever the
    # fixture was not installed, and pytest escalates that to `INTERNALERROR`,
    # taking the session down instead of the test. Reported reviewing PR #121,
    # round 8. `test_the_conftest_installs_every_hook_this_module_defines`
    # addresses the partial install that made it reachable.
    observed = item.stash.get(_OBSERVED, [])
    during_setup = observed[: item.stash.get(_CALL_START, 0)]
    during_call = observed[item.stash.get(_CALL_START, 0) :]

    missing = unearned(claims, during_call)
    if missing:
        note = ""
        if not unearned(missing, during_setup):
            note = (
                "\n\nThose rejections did happen — during **setup**, so a fixture "
                "caused them rather than this test. Drive the capability from the "
                "test body, or move the marker to a test that does."
            )
        _fail(
            report,
            f"unearned rejection marker(s): {missing}\n\n"
            "The marker claims this test proves a gate can say no, so the run has "
            "to show that gate reporting `passed=False`. It did not.\n"
            f"Verdicts observed while the body ran: {_summarise(during_call)}" + note,
        )
    return report
