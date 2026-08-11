"""Turn a run of unimproving experiments into a hypothesis that names a change.

`maybe_mint_improvement_hypothesis` already reacts when one execution loses to
its parent. What nothing reacted to is a campaign where each experiment looks
acceptable on its own and the score still has not moved — the case M8 exists
for. That needs the `ScoreEvent` series, which lives on `BudgetState`, so this
sits in the conductor rather than the execution layer that owns the
per-execution mint.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from labpilot.research_engine.conductor.budgets import (
    BudgetConfig,
    BudgetState,
    ScoreEvent,
    comparable_tail,
    score_summary,
)
from labpilot.research_engine.shared.labels import is_record_reference

if TYPE_CHECKING:
    from labpilot.research_engine.workspace_facade import Workspace

logger = logging.getLogger(__name__)


def techniques_in(events: list[ScoreEvent]) -> list[str]:
    """Every technique named anywhere in the window, combo members included.

    A technique already tried as one half of a combination has been tried, so
    proposing it alone as though it were untested would re-run work the window
    already covers. Record references (`hyp:H-010`, `fork:H-003`) travel in
    these fields as provenance and are not techniques — the rule for that lives
    in `shared.labels` because it was implemented wrong twice before.
    """
    names: list[str] = []
    for event in events:
        for name in [event.technique, *event.combo_techniques]:
            text = str(name or "").strip()
            if text and not is_record_reference(text) and text not in names:
                names.append(text)
    return names


def stagnation_window(state: BudgetState, config: BudgetConfig) -> list[ScoreEvent]:
    """The experiments that have run since the score last improved.

    Empty when the campaign is not stagnant. Read from the comparable tail, so
    readings from before a metric change never join a window whose whole point
    is that none of them beat the others.
    """
    summary = score_summary(state, config)
    window = max(1, config.plateau_window)
    if summary.steps_since_improvement < window:
        return []
    events = comparable_tail(state.score_events)
    return events[-summary.steps_since_improvement :] if summary.steps_since_improvement else []


def _untried_technique(workspace: Workspace, exclude: list[str]) -> str | None:
    """A technique worth proposing that the window has not already spent.

    Sourced from the experiment ledger's `techniques_untried` rather than
    `generate_candidates`. The design doc chose the latter on the grounds that
    it takes no `llm_client` — true of that function, but it requires a
    `ResearchContext` that only `ContextBuilder` builds, and `ContextBuilder`
    does take one. The ledger answers the narrower question actually being
    asked here, from `knowledge_dir` and `competition` alone.

    Filtered by the vocabulary's own planner-visible statuses, via the same
    `is_planner_visible` predicate `filter_by_technique_status` uses — that
    one takes `list[HypothesisCandidate]` rather than bare names, so it can't
    be called directly here, but the default-status policy underneath it
    should still live in one place. The ledger calls `list_techniques()`
    unfiltered and its `TechniqueRecord.status` is a different axis —
    worked/failed/untried, derived from hypotheses — so without this a
    proposal could name `the`, which is exactly the junk M18 exists to keep
    away from the planner.
    """
    from labpilot.research_engine.execution.technique.status_constants import (
        is_planner_visible,
    )
    from labpilot.research_engine.intelligence.hypothesis.ledger import build_experiment_ledger
    from labpilot.research_engine.intelligence.hypothesis.persist import load_open_hypothesis_tags
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    spent = {normalize_label(name) for name in exclude}
    try:
        ledger = build_experiment_ledger(workspace.knowledge_dir, workspace.competition)
        # A technique named by an open hypothesis is not untried — it is
        # queued. `techniques_untried` only excludes what a CONFIRMED or
        # REJECTED hypothesis marked worked or failed, so without this every
        # plateau in a campaign re-proposes the same name and grows the
        # backlog M21 exists because of.
        spent |= load_open_hypothesis_tags(workspace.knowledge_dir, workspace.competition)
        with KnowledgeStore(workspace.knowledge_dir, workspace.competition) as store:
            statuses = {
                normalize_label(str(row.get("name") or "")): str(row.get("status") or "candidate")
                for row in store.list_techniques()
            }
    except Exception:  # noqa: BLE001 — no proposal is better than a broken campaign
        logger.info("cannot read the technique inventory; no stagnation hypothesis minted")
        return None

    for name in ledger.techniques_untried:
        label = normalize_label(name)
        if label in spent or is_record_reference(name):
            continue
        if is_planner_visible(statuses.get(label)):
            return name
    return None


def _cite(event: ScoreEvent) -> str:
    """One experiment as the reason string names it.

    A combo is cited whole. Picking one member to blame for a delta produced
    by two or three together is the misattribution M19 §5 fixed for evidence
    cards, and it would be no more true here.
    """
    if event.combo_techniques:
        return f"{event.experiment_id} ({' + '.join(event.combo_techniques)})"
    if event.technique:
        return f"{event.experiment_id} ({event.technique})"
    return event.experiment_id


def _cite_list(events: list[ScoreEvent], *, limit: int = 12) -> str:
    """Every experiment named, unless there are too many to fit in prose.

    `reason`/`observation` used to be built by joining every citation and
    then hard-truncating the whole string to a length cap — for a plateau
    long enough to push the joined string past that cap, the cut landed
    mid-citation and silently dropped whichever ids came after it. Capping by
    *count* instead keeps every printed id whole; the ones left out are named
    by number, and the structured `evidence` list (never truncated) is still
    where a reader resolves every id, which is the guarantee that actually
    matters.
    """
    cites = [_cite(event) for event in events]
    if len(cites) <= limit:
        return ", ".join(cites)
    shown = ", ".join(cites[:limit])
    return f"{shown}, and {len(cites) - limit} more (see evidence)"


def mint_stagnation_hypothesis(
    workspace: Workspace,
    state: BudgetState,
    config: BudgetConfig,
) -> str | None:
    """Propose a change of direction, or None with the reason logged.

    Returns the new hypothesis id. Callers must have established that the
    campaign is stagnant *and* that no mint has fired for this plateau —
    firing on every step of a long one would flood the backlog with
    near-duplicates, which content dedup alone does not prevent.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import (
        HypothesisCreatedBy,
        HypothesisGenerator,
        HypothesisOrigin,
    )

    window = stagnation_window(state, config)
    if not window:
        return None

    summary = score_summary(state, config)
    spent = techniques_in(window)
    proposal = _untried_technique(workspace, spent)
    if proposal is None:
        # Naming no technique is the honest outcome when the inventory has
        # nothing left to suggest, and inventing one would put a fabricated
        # cause on the record. The operator still has the observation.
        logger.info(
            "%d experiments without improvement and no untried technique to propose; "
            "no stagnation hypothesis minted",
            len(window),
        )
        return None

    cited = _cite_list(window)
    best = summary.best_so_far
    metric = summary.metric_name or "the primary metric"
    observation = (
        f"{len(window)} experiments since {metric} last improved: {cited}. Best remains {best}."
    )
    reason = (
        f"None of {cited} improved on {metric}={best}, so the pattern they share "
        f"is not what is holding the score back."
    )
    prediction = (
        f"Testing {proposal}, which none of those experiments used, moves {metric} "
        f"where repeating their approach has not."
    )

    store = HypothesisStore(workspace.knowledge_dir, workspace.competition)
    try:
        minted = store.create(
            observation=observation[:500],
            reason=reason[:1000],
            prediction=prediction[:500],
            confidence=0.5,
            tags=["stagnation", *(e.experiment_id for e in window)],
            source="reflection",
            created_by=HypothesisCreatedBy.REFLECTION,
            generator=HypothesisGenerator.RULE_ENGINE,
            origin=HypothesisOrigin.EXPERIMENT,
            # One ref per experiment, not just the newest: a reader checking
            # "did this cite its evidence" must be able to resolve every id.
            evidence=[
                {
                    "kind": "experiment",
                    "ref": event.experiment_id,
                    "note": f"{metric}={event.value}, no improvement",
                }
                for event in window
            ],
            technique=proposal,
        )
    except Exception:  # noqa: BLE001 — a failed mint must not end the campaign
        logger.exception("could not mint a stagnation hypothesis")
        return None

    logger.info(
        "minted %s after %d experiments without improvement; proposes %s",
        minted.id,
        len(window),
        proposal,
    )
    return minted.id
