"""Which hypotheses still count as work worth doing.

`untested_hypothesis_count` counted every `proposed` row, and that number is the
one thing standing between a campaign and fresh evidence. Measured on rogii
2026-08-09: **46 proposed**, of which 43 were generated on 2026-08-07 and never
selected — including `3D garment modeling` and `Breath Focus practice` for a
wellbore-geology regression. Each held the fetch gate shut exactly as firmly as
a good idea would.

So the pool defends itself: the only thing that would refresh it is disabled by
its own size, and size is the one property that says nothing about quality.

### Stale means never chosen, not merely old

A hypothesis proposed long ago that the campaign has *repeatedly declined to
select* is not a queue of pending work — it is a queue the selector has already
rejected in practice, silently, once per step.

That is the same judgement M18 applies to techniques: `derive_technique_status`
retires one that was never measured, never selected, and has outlived
``DORMANT_AFTER_CAMPAIGNS``. Reusing the rule rather than inventing a second one
keeps a single answer to "is this still live?", and reuses `campaigns_since`
rather than wall-clock time — a workspace idle for a week has not declined
anything.

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

#: Campaigns a never-selected hypothesis may sit through before it stops
#: counting as pending work. Matches `DORMANT_AFTER_CAMPAIGNS`, deliberately:
#: two independent staleness clocks would drift and disagree about the same
#: workspace.
STALE_AFTER_CAMPAIGNS = 2


def viable_hypothesis_count(knowledge_dir: Path, competition: str) -> int:
    """`proposed` hypotheses that still represent work the campaign might do.

    Excludes those the selector has passed over for `STALE_AFTER_CAMPAIGNS`
    campaigns without ever planning them. Never raises: an unreadable store
    means "nothing queued", which opens the gate rather than closing it — the
    failure this whole module exists to prevent is a gate stuck shut.
    """
    from labpilot.research_engine.execution.technique.vocabulary import (
        campaign_created_ats,
    )
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import HypothesisStatus

    try:
        store = HypothesisStore(Path(knowledge_dir), competition)
        proposed = store.list(status=HypothesisStatus.PROPOSED)
    except Exception:  # noqa: BLE001 — absent store means nothing queued
        return 0
    if not proposed:
        return 0

    try:
        sessions = tuple(campaign_created_ats(Path(knowledge_dir), competition))
    except Exception:  # noqa: BLE001 — without a clock nothing can be stale
        return len(proposed)

    return sum(1 for hypothesis in proposed if not _is_stale(hypothesis, sessions))


def _is_stale(hypothesis: object, sessions: tuple) -> bool:
    """True when the selector has had chances to pick this and never did.

    `evidence_for` / `evidence_against` being empty is the proxy for "never
    planned": a hypothesis that reached a plan gets `testing` and leaves
    `proposed` entirely, so anything still here with no evidence has never been
    run.
    """
    from labpilot.research_engine.execution.technique.vocabulary import campaigns_since

    if getattr(hypothesis, "evidence_for", None) or getattr(hypothesis, "evidence_against", None):
        return False
    try:
        age = campaigns_since(getattr(hypothesis, "created_at", None), sessions)
    except (TypeError, ValueError) as exc:
        # Narrow, and logged. A blanket `except Exception: return False` here
        # swallowed a real `TypeError` — campaign stamps are timezone-aware and
        # `Hypothesis.created_at` is naive — and turned this filter into a
        # silent no-op: 46 rows in, 46 out, on a workspace where 43 were stale.
        # A broken guard that reports healthy is the failure this project keeps
        # paying for, so an unexpected shape says so rather than passing.
        logger.warning("could not age hypothesis; treating as live: %s", exc)
        return False
    return age >= STALE_AFTER_CAMPAIGNS
