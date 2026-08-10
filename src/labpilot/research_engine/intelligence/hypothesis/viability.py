"""Which hypotheses still count as work worth doing.

`untested_hypothesis_count` counted every `proposed` row, and that number is the
one thing standing between a campaign and fresh evidence. Measured on rogii
2026-08-09: **46 proposed**, of which 43 were generated on 2026-08-07 and never
selected — including `3D garment modeling` and `Breath Focus practice` for a
wellbore-geology regression. Each held the fetch gate shut exactly as firmly as
a good idea would.

So the pool defends itself: the only thing that would refresh it is disabled by
its own size, and size is the one property that says nothing about quality.

### Stale means passed over, not merely old

A hypothesis is stale when the selector had chances to pick it and picked
something else. That is measured in **selections** — plans minted against some
other hypothesis — not in campaigns started and not in wall-clock time.

Campaign count was the first attempt and it is too generous: a campaign that
crashed at step three never chose anything, so counting it as a rejection
punishes a hypothesis for an infrastructure failure. On rogii, 37 campaigns had
run and many were the failing ones, which is why every hypothesis aged out at
once.

Wall-clock is worse still: a workspace idle for a week has declined nothing.

Counting selections says exactly what the word means — *this was available N
times and passed over N times*.

**Counting, not deleting.** Nothing here changes a hypothesis's status. A stale
row stays exactly where it is and can still be selected if it ranks; it simply
stops voting on whether the campaign is allowed to learn something new. The
distinction matters because this project has retired real findings by accident
before, and because the fix for a bad backlog is better evidence, not amnesia.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Times a hypothesis may be passed over before it stops counting as pending
#: work. Matches `DORMANT_AFTER_CAMPAIGNS` so the two staleness rules agree
#: about the same workspace rather than drifting apart.
STALE_AFTER_SELECTIONS = 2


def viable_hypothesis_count(knowledge_dir: Path, competition: str) -> int:
    """`proposed` hypotheses that still represent work the campaign might do.

    Excludes those the selector has passed over `STALE_AFTER_SELECTIONS` times.
    Never raises: an unreadable store means "nothing queued", which opens the
    gate rather than closing it — the failure this module exists to prevent is a
    gate stuck shut.
    """
    from labpilot.research_engine.intelligence.paths import hypotheses_are_absent
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    if hypotheses_are_absent(knowledge_dir, competition):
        return 0
    try:
        store = HypothesisStore(Path(knowledge_dir), competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:
        # This count opens M21's gathering gate. A fault reporting zero is the
        # gate stuck shut — the failure this module's own docstring exists to
        # prevent, arriving through its error path. M20, 2026-08-09.
        logger.exception("cannot count viable hypotheses for %s; treating as none", competition)
        return 0
    return _viable_of(Path(knowledge_dir), competition, proposed)


def _viable_of(knowledge_dir: Path, competition: str, proposed: list) -> int:
    """The filter itself, over rows the caller has already read."""
    if not proposed:
        return 0

    selections = _selection_times(Path(knowledge_dir), competition)
    if not selections:
        # Nothing has ever been selected, so nothing has been passed over.
        return len(proposed)

    try:
        return sum(1 for hypothesis in proposed if not _is_stale(hypothesis, selections))
    except Exception as exc:  # noqa: BLE001 — the contract above is "never raises"
        # The docstring's promise was only half kept: the store read was
        # guarded and the count was not. `_is_stale` narrowed its own catch to
        # (TypeError, ValueError) for good reason, which left anything else —
        # an AttributeError off a malformed row — to propagate through
        # `should_gather_evidence` into the policy step and end the campaign.
        #
        # That is the same failure this module exists to remove, reached by
        # crashing instead of by lying. Failing open means the whole pool
        # counts, which at worst delays a fetch; failing closed stops the run.
        logger.warning("viability filter failed; counting the whole pool: %s", exc)
        return len(proposed)


def pool_counts(knowledge_dir: Path, competition: str) -> tuple[int, int]:
    """`(viable, proposed_total)` from **one** read of the store.

    Both numbers are wanted together — "46 proposed, 3 viable" says more than
    either alone, and the gap is itself the signal that the pool has gone stale
    — and asking for them separately meant globbing every `H-*.json`, parsing
    each and mirroring the whole pool into SQLite twice, back to back, on every
    policy step.

    `should_gather_evidence` still asks for the viable count on its own, so a
    step reads the store twice rather than three times. Removing the last one
    needs a per-step context object to hang it on; `build_observe_bundle` and
    `available_tools` are called separately by `decide_next` and share nothing
    today.
    """
    from labpilot.research_engine.intelligence.paths import hypotheses_are_absent
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    if hypotheses_are_absent(knowledge_dir, competition):
        return 0, 0
    try:
        store = HypothesisStore(Path(knowledge_dir), competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:
        # Both numbers, because both are read here. See `viable_hypothesis_count`.
        logger.exception("cannot read the hypothesis pool for %s; reporting none", competition)
        return 0, 0
    return _viable_of(Path(knowledge_dir), competition, proposed), len(proposed)


def _selection_times(knowledge_dir: Path, competition: str) -> tuple[tuple[object, str], ...]:
    """`(when, hypothesis_id)` for every selection, oldest first.

    A plan carrying a `hypothesis_id` is a selection: that is the moment one
    idea was preferred over every other open one.

    The id is carried alongside the timestamp so `_is_stale` can drop a
    hypothesis's own selections before aging it.
    """
    from labpilot.research_engine.execution.technique.vocabulary import _parse_timestamp
    from labpilot.research_engine.planner.store import PlanStore

    store = PlanStore(Path(knowledge_dir), competition)
    try:
        pairs = [
            (_parse_timestamp(created_at), hypothesis_id)
            for created_at, hypothesis_id in store.hypothesis_selection_times()
        ]
    except Exception:  # noqa: BLE001 — no plans means nothing was ever chosen
        return ()
    finally:
        store.close()
    return tuple(sorted((s, h) for s, h in pairs if s is not None))


def _is_stale(hypothesis: object, selections: tuple[tuple[object, str], ...]) -> bool:
    """True when the selector chose *something else* `STALE_AFTER_SELECTIONS` times.

    Something else, and that word is the fix. A hypothesis's own selections are
    dropped before aging it, exactly as `vocabulary.derive_technique_status`
    excludes a technique's own via its `selected` set.

    Without that, a hypothesis was aged by its own retries. `record_failed_
    attempt` returns a `RETRYABLE` failure to `proposed` with no evidence
    written, so H-001 selected at t1, failed on a rate limit, and back in the
    pool would count t1 against itself; one more selection by anyone else and it
    was stale after a single transient failure — out of `viable_hypothesis_
    count`, with two of its three attempts unused, thinning the very pool the
    count gates fetching on.

    `evidence_for` / `evidence_against` being empty is the proxy for "no
    measurement yet". It is not a proxy for "never planned": the retry path
    above puts planned-but-unmeasured rows back here, which is precisely why the
    id filter is needed rather than assumed unnecessary.
    """
    from labpilot.research_engine.execution.technique.vocabulary import campaigns_since

    if getattr(hypothesis, "evidence_for", None) or getattr(hypothesis, "evidence_against", None):
        return False
    own_id = str(getattr(hypothesis, "id", "") or "")
    others = tuple(stamp for stamp, hypothesis_id in selections if hypothesis_id != own_id)
    if not others:
        return False
    try:
        age = campaigns_since(getattr(hypothesis, "created_at", None), others)
    except (TypeError, ValueError) as exc:
        # Narrow, and logged. A blanket `except Exception: return False` here
        # swallowed a real `TypeError` — campaign stamps are timezone-aware and
        # `Hypothesis.created_at` is naive — and turned this filter into a
        # silent no-op: 46 rows in, 46 out, on a workspace where 43 were stale.
        # A broken guard that reports healthy is the failure this project keeps
        # paying for, so an unexpected shape says so rather than passing.
        logger.warning("could not age hypothesis; treating as live: %s", exc)
        return False
    return age >= STALE_AFTER_SELECTIONS
