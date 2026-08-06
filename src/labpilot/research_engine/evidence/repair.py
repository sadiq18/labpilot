"""Re-orient evidence cards that were written with the wrong metric direction.

Cards built before the direction fix recorded ``maximize=True`` regardless of the
competition. On a minimised metric that reverses every verdict, and the verdicts
are what the Conductor plans against — so leaving them in place is not a cosmetic
debt, it is the campaign steering by an inverted compass.

Repair runs from the campaign, not as a one-off migration script the user must
remember. That is the same principle as `ClaimPromoter.revalidate_claims`:
correct what is already recorded before adding to it. It also means a workspace
whose profile is fixed later heals on the next run without anyone editing stored
artifacts by hand.

Only ``decision``, ``decision_reason``, ``claim_updates`` and ``maximize`` are
rewritten, all recomputed from the scores already on the card. Measurements are
never touched: what was observed does not change, only what it means.
"""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.evidence.builder import (
    _claim_updates_from_attribution,
    decide_evidence,
    is_placeholder_metrics,
)
from labpilot.research_engine.evidence.models import EvidenceDecision
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.competition.direction import resolve_maximize
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.paths import ResearchPaths

logger = logging.getLogger(__name__)


def _placeholder_executions(knowledge_dir: Path, competition: str) -> set[str]:
    """Execution ids whose stored metrics say no model was trained.

    Legacy cards do not record which run produced them beyond an id, but the
    execution artifacts still hold the original metrics — including the
    ``dry_run_stub`` / ``last_resort_scaffold`` status. So a card minted before
    the placeholder guard can still be identified after the fact, from evidence
    already on disk rather than from a guess about its scores.
    """
    found: set[str] = set()
    try:
        with KnowledgeStore(Path(knowledge_dir), competition) as store:
            for artifact in store.list_artifacts():
                meta = artifact.metadata or {}
                exec_id = str(meta.get("execution_id") or "")
                if exec_id and is_placeholder_metrics(meta.get("metrics")):
                    found.add(exec_id)
    except Exception as exc:  # noqa: BLE001 — repair must never break a run
        logger.debug("could not scan executions for placeholders: %s", exc)
    return found


def repair_card_directions(
    knowledge_dir: Path, competition: str, *, workspace_root: Path | None = None
) -> list[str]:
    """Rewrite verdicts on cards whose stored direction is wrong. Returns their ids.

    A no-op when the direction cannot be resolved: without a known direction
    there is nothing to correct *towards*, and rewriting on a guess would be the
    original defect wearing a different hat.
    """
    paths = ResearchPaths(Path(knowledge_dir), competition)
    maximize = resolve_maximize(
        competition=competition,
        workspace_root=workspace_root,
        knowledge_root=paths.root,
        extracted_dir=paths.extracted_dir,
    )
    if maximize is None:
        logger.debug("no metric direction for %s; not repairing cards", competition)
        return []

    store = EvidenceCardStore(Path(knowledge_dir), competition)
    try:
        cards = store.list()
    except Exception as exc:  # noqa: BLE001 — repair must never break a run
        logger.warning("could not list evidence cards for repair: %s", exc)
        return []

    placeholders = _placeholder_executions(knowledge_dir, competition)

    repaired: list[str] = []
    for card in cards:
        # Retire cards built from runs that never trained. Checked before the
        # direction test because such a card has no meaningful verdict to
        # re-orient: correcting its sign would only make a fabricated
        # comparison point the other way.
        from_placeholder = (
            card.treatment_experiment in placeholders or card.control_experiment in placeholders
        )
        if from_placeholder and card.decision != EvidenceDecision.INCONCLUSIVE:
            updated = card.model_copy(
                update={
                    "maximize": maximize,
                    "decision": EvidenceDecision.INCONCLUSIVE,
                    "decision_reason": (
                        "placeholder_metrics: run reported no trained model (retired by repair)"
                    ),
                    "claim_updates": [],
                }
            )
            try:
                store.save(updated)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not retire card %s: %s", card.id, exc)
                continue
            logger.info("Retired %s: built from a placeholder run", card.id)
            repaired.append(card.id)
            continue
        if from_placeholder:
            continue
        if bool(card.maximize) == maximize:
            continue
        decision, reason = decide_evidence(
            cv_gain=card.observed.cv_gain,
            lb_gain=card.observed.lb_gain,
            stability=card.observed.stability,
            maximize=maximize,
            missing_control=card.control_experiment is None and card.observed.parent_cv is None,
        )
        updated = card.model_copy(
            update={
                "maximize": maximize,
                "decision": decision,
                "decision_reason": f"{reason} (re-oriented: direction was inverted)",
                "claim_updates": _claim_updates_from_attribution(
                    dict(card.technique_attribution or {}),
                    decision=decision,
                    maximize=maximize,
                ),
            }
        )
        try:
            store.save(updated)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not repair card %s: %s", card.id, exc)
            continue
        logger.info(
            "Re-oriented %s: %s -> %s (maximize %s -> %s)",
            card.id,
            card.decision,
            decision,
            card.maximize,
            maximize,
        )
        repaired.append(card.id)
    return repaired
