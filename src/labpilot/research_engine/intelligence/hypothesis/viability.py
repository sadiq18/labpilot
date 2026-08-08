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
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    try:
        store = HypothesisStore(Path(knowledge_dir), competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:  # noqa: BLE001 — absent store means nothing queued
        return 0
    if not proposed:
        return 0

    selections = _selection_times(Path(knowledge_dir), competition)
    if not selections:
        # Nothing has ever been selected, so nothing has been passed over.
        return len(proposed)

    return sum(1 for hypothesis in proposed if not _is_stale(hypothesis, selections))


def retired_hypothesis_ids(knowledge_dir: Path, competition: str) -> set[str]:
    """Hypotheses settled as `rejected` — nothing further to learn from them.

    Needed because the campaign selects **plans**, not hypotheses, and the two
    retire independently. Measured on rogii 2026-08-09: redundancy detection
    correctly rejected `H-051`, and the very next step selected `P-021` again —
    the plan carrying it, still `in_progress` and therefore still runnable.

    Retiring the idea has to retire the work queued against it, or the loop the
    retirement exists to break simply continues one level up.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    try:
        store = HypothesisStore(Path(knowledge_dir), competition)
        return {h.id for h in store.list(status=HypothesisStatus.REJECTED)}
    except Exception:  # noqa: BLE001 — unreadable store retires nothing
        return set()


def plan_is_selectable(plan: object, retired: set[str]) -> bool:
    """False when this plan tests an idea already retired.

    A plan with no hypothesis — a baseline — is always selectable: there is no
    retired idea behind it.
    """
    hypothesis_id = str(getattr(plan, "hypothesis_id", "") or "")
    return not hypothesis_id or hypothesis_id not in retired


def _selection_times(knowledge_dir: Path, competition: str) -> tuple:
    """When the selector chose *some* hypothesis, oldest first.

    A plan carrying a `hypothesis_id` is a selection: that is the moment one
    idea was preferred over every other open one.
    """
    from labpilot.research_engine.execution.technique.vocabulary import _parse_timestamp
    from labpilot.research_engine.planner.store import PlanStore

    store = PlanStore(Path(knowledge_dir), competition)
    try:
        stamps = [
            _parse_timestamp(plan.created_at)
            for plan in store.list_plans()
            if getattr(plan, "hypothesis_id", None)
        ]
    except Exception:  # noqa: BLE001 — no plans means nothing was ever chosen
        return ()
    finally:
        store.close()
    return tuple(sorted(s for s in stamps if s is not None))


def _is_stale(hypothesis: object, selections: tuple) -> bool:
    """True when the selector chose something else `STALE_AFTER_SELECTIONS` times.

    `evidence_for` / `evidence_against` being empty is the proxy for "never
    planned": a hypothesis that reached a plan gets `testing` and leaves
    `proposed` entirely, so anything still here with no evidence has never been
    run.
    """
    from labpilot.research_engine.execution.technique.vocabulary import campaigns_since

    if getattr(hypothesis, "evidence_for", None) or getattr(hypothesis, "evidence_against", None):
        return False
    try:
        age = campaigns_since(getattr(hypothesis, "created_at", None), selections)
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
