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
    ScoreSummary,
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


def stagnation_window(
    state: BudgetState, config: BudgetConfig, *, summary: ScoreSummary | None = None
) -> list[ScoreEvent]:
    """The experiments that have run since the score last improved.

    Empty when the campaign is not stagnant. Read from the comparable tail, so
    readings from before a metric change never join a window whose whole point
    is that none of them beat the others.

    `summary` lets a caller that already has one (it computes `score_summary`
    internally either way) skip a second `comparable_tail` scan. Optional so
    direct callers (tests) keep working unchanged.
    """
    if summary is None:
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


_MORE_NOTE_RESERVE = len(", and 999 more (see evidence)")


def _cite_list(events: list[ScoreEvent], *, limit: int = 12, max_chars: int = 400) -> str:
    """Every experiment named, unless there are too many to fit in `max_chars`.

    `reason`/`observation` used to be built by joining every citation and
    then hard-truncating the *whole assembled string* (wrapper text and all)
    to a length cap — for a plateau long enough, or wrapper text (the metric
    name) long enough, to push the joined string past that cap, the cut
    landed mid-citation and silently dropped whichever ids came after it.
    Reproduced live twice: once with 12 two-technique combo citations alone,
    and again with a merely-realistic metric name added on top of a fixed
    per-citation char budget — the wrapper text sat outside that budget
    entirely. `max_chars` here must be sized by the caller to what's actually
    left over after its own wrapper text, not to the field's raw cap.

    The trailing "and N more" note's own length is reserved up front, not
    just the citations' — appending it after an unbudgeted citation list
    would silently blow the cap the same way. The ones left out are named by
    number; the structured `evidence` list (never truncated) is still where a
    reader resolves every id, which is the guarantee that actually matters.
    """
    cites = [_cite(event) for event in events]
    budget = max_chars - _MORE_NOTE_RESERVE
    shown: list[str] = []
    joined = ""
    for cite in cites[:limit]:
        candidate = f"{joined}, {cite}" if joined else cite
        if shown and len(candidate) > budget:
            break
        joined = candidate
        shown.append(cite)
    remaining = len(cites) - len(shown)
    if remaining <= 0:
        return joined
    return f"{joined}, and {remaining} more (see evidence)"


def mint_stagnation_hypothesis(
    workspace: Workspace,
    state: BudgetState,
    config: BudgetConfig,
    *,
    window: list[ScoreEvent] | None = None,
    summary: ScoreSummary | None = None,
) -> str | None:
    """Propose a change of direction, or None with the reason logged.

    Returns the new hypothesis id. Callers must have established that the
    campaign is stagnant *and* that no mint has fired for this plateau —
    firing on every step of a long one would flood the backlog with
    near-duplicates, which content dedup alone does not prevent.

    `window` lets a caller that already computed `stagnation_window` (to
    decide whether to call this at all) pass it straight through instead of
    this function deriving it again — `_maybe_mint_on_stagnation` runs on
    every recorded experiment, stagnant or not, so recomputing the window's
    `comparable_tail`/`score_summary` scan a second time here was paid on
    every step for no reason. `summary` is the same trade for the same
    reason: `stagnation_window` already computes one internally to decide
    the window's size, so a caller that has *that* summary in hand can skip
    this function's own `score_summary` call too — the first cut of this
    optimization only threaded `window` and missed that `score_summary` was
    still running unconditionally right below it. Both left optional so
    direct callers (tests) keep working unchanged.
    """
    from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
    from labpilot.research_engine.shared.experiments.models import (
        HypothesisCreatedBy,
        HypothesisGenerator,
        HypothesisOrigin,
    )

    if window is None:
        window = stagnation_window(state, config)
    if not window:
        return None

    if summary is None:
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

    best = summary.best_so_far
    metric = summary.metric_name or "the primary metric"

    def _observation(cited: str) -> str:
        return (
            f"{len(window)} experiments since {metric} last improved: {cited}. Best remains {best}."
        )

    def _reason(cited: str) -> str:
        return (
            f"None of {cited} improved on {metric}={best}, so the pattern they share "
            f"is not what is holding the score back."
        )

    # `_cite_list`'s budget must be sized to what's actually left after each
    # template's own wrapper text (dominated by `metric`, which has no length
    # bound) -- a fixed citation budget alone still let a long metric name
    # push the assembled string past its cap, cutting a citation mid-word.
    observation = _observation(_cite_list(window, max_chars=max(0, 500 - len(_observation("")))))
    reason = _reason(_cite_list(window, max_chars=max(0, 1000 - len(_reason("")))))
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
