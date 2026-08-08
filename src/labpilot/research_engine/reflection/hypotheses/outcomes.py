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
from enum import StrEnum

logger = logging.getLogger(__name__)

#: Failure kinds that say nothing about the hypothesis itself. The provider was
#: busy, the model timed out, the endpoint was down — none of which is evidence
#: about the idea being tested.
_TRANSIENT_KINDS: frozenset[str] = frozenset({"rate_limit", "unavailable", "timeout", "no_client"})

#: Attempts before an otherwise-retryable hypothesis is retired.
#:
#: Three, matching `llm_max_attempts`. The fourth identical failure teaches
#: nothing the third did not, and every attempt after it is a campaign step
#: spent on an idea that has not worked yet.
DEFAULT_MAX_ATTEMPTS = 3


class HypothesisOutcome(StrEnum):
    """What to do with a hypothesis whose experiment failed."""

    #: Infrastructure failed under it. Return it to the pool.
    RETRYABLE = "retryable"
    #: Trying again cannot help. Retire it, with the reason.
    DEAD_END = "dead_end"


def classify_hypothesis_failure(
    *,
    failure_reason: str = "",
    failure_kind: str | None = None,
    attempts: int = 1,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
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

    if attempts >= max_attempts:
        return (
            HypothesisOutcome.DEAD_END,
            f"failed {attempts} time(s), the last as {failure_kind or 'unknown'}: "
            f"{failure_reason or 'no reason recorded'}",
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
