"""Did this hypothesis fail, or did the infrastructure fail under it?

The distinction the campaign needs and did not have. Measured on rogii
2026-08-09: a hypothesis leaves `testing` **only** when an evidence card is
written, and a failed execution writes no card — so a hypothesis whose
experiment failed stayed `testing` forever, out of the pool but never retired.
One is stuck now; three were stuck historically.

Meanwhile a hypothesis whose change was already implemented stayed `proposed`
and was re-selected on every step of four consecutive campaigns.

Both are the same missing idea: **a failed attempt has to say whether trying
again could ever help.**

### The rule is the one the LLM path already follows

`BaseMicroAgent.run` retries transient failures and records a failure only when
attempts are exhausted — a call that succeeds on attempt 2 is a success, not a
failure with a caveat. This is that rule one layer up, and it reuses the same
vocabulary: `failure_kind` already separates transient (`rate_limit`,
`unavailable`, `timeout`) from the rest.

### Why attempts are derived, not stored

The count comes from the executions that already exist, rather than a counter on
the hypothesis. A stored counter is a derived value that drifts from its source
the moment anything writes one and not the other — the failure this project has
now fixed three times (plan projections, evidence-card dumps, skill overlays).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import StrEnum

from labpilot.accessor.common.provenance import failure_signature

logger = logging.getLogger(__name__)

#: Failure kinds that say nothing about the hypothesis itself. The provider was
#: busy, the model timed out, the endpoint was down — none of which is evidence
#: about the idea being tested.
_TRANSIENT_KINDS: frozenset[str] = frozenset({"rate_limit", "unavailable", "timeout", "no_client"})

#: Attempts before an otherwise-retryable hypothesis is retired **on a repeat**.
#:
#: Three, matching `llm_max_attempts`. The fourth identical failure teaches
#: nothing the third did not, and every attempt after it is a campaign step
#: spent on an idea that has not worked yet.
#:
#: **Identical** was load-bearing and only the prose knew it (issue #176, the
#: sibling of #173 one layer down). Three attempts that each fix the previous
#: defect and surface a new one is the repair loop converging, and retiring on
#: that wrote `REJECTED` — a durable claim about the *idea* — on evidence that
#: only said the generated code did not run yet.
DEFAULT_MAX_ATTEMPTS = 3

#: The ceiling regardless of whether the failures repeat.
#:
#: The campaign breaker could hand its distinct-failure case to
#: `max_barren_steps`; there is no equivalent here, and a hypothesis whose
#: failures are endlessly novel would never retire at all — a worse failure
#: than retiring one early, because the selector would keep offering it.
#:
#: Eight to three is the ratio the campaign layer already uses
#: (`DEFAULT_MAX_BARREN_STEPS` over `DEFAULT_MAX_CONSECUTIVE_FAILURES`), for the
#: same reason: the slower limit is not a second opinion about the same
#: question, it is the backstop for the case the faster one declines to judge.
DEFAULT_MAX_DISTINCT_ATTEMPTS = 8


class HypothesisOutcome(StrEnum):
    """What to do with a hypothesis whose experiment failed."""

    #: Infrastructure failed under it. Return it to the pool.
    RETRYABLE = "retryable"
    #: Trying again cannot help. Retire it, with the reason.
    DEAD_END = "dead_end"


def _failures_are_repeating(recent_failures: Sequence[str]) -> bool:
    """True when this hypothesis is stuck rather than working through defects.

    The newest failure against every other one still recorded, not against the
    previous one alone: a loop where fixing A reintroduces B and fixing B
    reintroduces A differs from its predecessor every time and is a stall by any
    reading. `BudgetState.failures_are_repeating` asks the same question of the
    campaign, and both defer to `failure_signature` so there is one answer to it.

    **Fewer than two recorded failures answers True**, keeping the old
    behaviour wherever this cannot see — a caller that supplies no history gets
    exactly the retirement it got before.
    """
    texts = [t for t in recent_failures if str(t).strip()]
    if len(texts) < 2:
        return True
    newest = failure_signature(texts[-1])
    return any(failure_signature(prior) == newest for prior in texts[:-1])


def classify_hypothesis_failure(
    *,
    failure_reason: str = "",
    failure_kind: str | None = None,
    attempts: int = 1,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_distinct_attempts: int = DEFAULT_MAX_DISTINCT_ATTEMPTS,
    recent_failures: Sequence[str] = (),
    redundant: bool = False,
) -> tuple[HypothesisOutcome, str]:
    """Return ``(outcome, why)`` for one failed attempt at a hypothesis.

    Order matters. Redundancy is checked first and settles it immediately: a
    change already present in the parent will be present on every future
    attempt, so retrying is guaranteed waste — this is the case that cost four
    campaigns.

    Exhaustion is checked before transience, deliberately. A rate limit is
    transient, and a rate limit that has blocked three attempts is still a
    campaign making no progress; treating "transient" as "retry forever" is how
    a loop with a plausible excuse runs to its step budget.
    """
    if redundant:
        return (
            HypothesisOutcome.DEAD_END,
            failure_reason or "the parent already implements this change",
        )

    if attempts >= max_distinct_attempts:
        return (
            HypothesisOutcome.DEAD_END,
            f"failed {attempts} time(s), the last as {failure_kind or 'unknown'}: "
            f"{failure_reason or 'no reason recorded'}",
        )

    # Exhausted *and* stuck. A hypothesis whose attempts each surfaced a new
    # defect is not evidence against the idea, and `RETRYABLE` below returns it
    # to the pool rather than writing a verdict the selector will honour.
    if attempts >= max_attempts and _failures_are_repeating(recent_failures):
        return (
            HypothesisOutcome.DEAD_END,
            f"failed {attempts} time(s) with the same failure, the last as "
            f"{failure_kind or 'unknown'}: {failure_reason or 'no reason recorded'}",
        )

    kind = str(failure_kind or "").strip().lower()
    if kind in _TRANSIENT_KINDS:
        return (
            HypothesisOutcome.RETRYABLE,
            f"attempt {attempts} hit a transient {kind}; the hypothesis is untested",
        )

    # Unknown failures are retryable *within* the attempt budget. An unrecognised
    # error is not evidence against the idea, and retiring on one would discard
    # hypotheses for defects in the harness — which this system has done before,
    # recording `SWA` as harmful when the fault was a metric direction.
    return (
        HypothesisOutcome.RETRYABLE,
        f"attempt {attempts} failed ({kind or 'unclassified'}); "
        "not yet evidence about the hypothesis",
    )
