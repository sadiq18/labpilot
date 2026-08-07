"""Re-derive beliefs from the evidence cards that currently exist.

`apply_card_to_beliefs` steps a belief once, as each card is written. That is
correct while cards are correct, and unrecoverable when they are not: repairing a
card afterwards does not retract the step it already caused. Measured on rogii
2026-08-07, `repair_card_directions` re-oriented all 15 cards and changed **zero**
beliefs — leaving `SWA`, the only technique that ever improved the metric,
recorded as `negative`.

So this module recomputes rather than un-does. Every evidence-derived belief is
rebuilt from the full current card set, which makes the result a function of the
cards alone: idempotent, and correct after any card repair without needing to
know what the cards used to say.

Two things it deliberately does **not** touch:

* Beliefs the KnowledgeHub wrote from literature. Those record how often a
  technique is *mentioned*, which is a different quantity from what it *did*
  here. They are merged for identity (below) but their confidence is not
  reinterpreted as evidence.
* Cards themselves. Direction repair owns those; this runs after it.

### The identity merge

Two writers used two id schemes for the same thing — `KnowledgeHub` writes
``belief_tech_<name>`` and `apply_card_to_beliefs` writes
``belief:<competition>:<slug>``. On rogii five techniques had *both*, disagreeing:
`belief_tech_swa` said `unknown` at 0.95 while `belief:...:swa` said `negative` at
0.38. Consumers iterate `list_beliefs()` and key by technique, so which one wins
is an ordering accident. The literature row won for `vit`, and at 0.95 it cleared
the promotion threshold.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.models import ClaimEvidenceKind, EvidenceDecision
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

logger = logging.getLogger(__name__)

#: Where a belief with no evidence sits. Matches `apply_card_to_beliefs`.
_NEUTRAL = 0.5

#: Marks a belief this module owns, so a rebuild can tell its own rows from
#: literature-derived ones without guessing from the id scheme.
_SOURCE_KEY = "confidence_source"
_SOURCE_EVIDENCE = "evidence_cards"


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "-").replace("/", "-").replace("+", "-")[:64]


def _metadata(belief: dict[str, Any]) -> dict[str, Any]:
    raw = belief.get("metadata")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _is_evidence_derived(belief: dict[str, Any]) -> bool:
    meta = _metadata(belief)
    return bool(meta.get("last_evidence_card_id")) or meta.get(_SOURCE_KEY) == _SOURCE_EVIDENCE


def _belief_status(confidence: float, kind: ClaimEvidenceKind | None) -> str:
    if kind == ClaimEvidenceKind.CONTRADICT and confidence <= 0.15:
        return "rejected"
    if confidence >= 0.7:
        return "validated"
    return "suggested"


def rederive_beliefs_from_cards(knowledge_dir: Path, competition: str) -> list[str]:
    """Rebuild evidence-derived beliefs from current cards. Returns changed ids.

    A technique that no longer has any conclusive card is reset to neutral
    rather than deleted — `vit` should stop being a 0.62 `positive` once the
    card behind it is retired, but the fact that we once believed it is part of
    the record.
    """
    try:
        cards = EvidenceCardStore(Path(knowledge_dir), competition).list()
    except Exception as exc:  # noqa: BLE001 — repair must never break a run
        logger.warning("could not read evidence cards for belief repair: %s", exc)
        return []

    # Accumulate from scratch over every card, so the result depends only on
    # what the cards say now.
    totals: dict[str, float] = {}
    kinds: dict[str, ClaimEvidenceKind] = {}
    last_card: dict[str, str] = {}
    for card in cards:
        if card.decision == EvidenceDecision.INCONCLUSIVE:
            continue
        for upd in card.claim_updates or []:
            tech = (upd.technique or "").strip()
            if not tech:
                continue
            totals[tech] = totals.get(tech, 0.0) + float(upd.confidence_delta)
            kinds[tech] = upd.evidence
            last_card[tech] = card.id

    changed: list[str] = []
    try:
        with KnowledgeStore(Path(knowledge_dir), competition) as store:
            existing = store.list_beliefs()
            by_technique: dict[str, list[dict[str, Any]]] = {}
            for belief in existing:
                by_technique.setdefault(str(belief.get("technique") or "").strip(), []).append(
                    belief
                )

            techniques = set(totals) | {
                str(b.get("technique") or "").strip()
                for b in existing
                if _is_evidence_derived(b)
            }
            for tech in sorted(t for t in techniques if t):
                canonical = f"belief:{competition}:{_slug(tech)}"
                rows = by_technique.get(tech, [])
                confidence = max(0.05, min(0.99, _NEUTRAL + totals.get(tech, 0.0)))
                kind = kinds.get(tech)
                effect = (
                    "positive"
                    if kind == ClaimEvidenceKind.SUPPORT
                    else "negative"
                    if kind == ClaimEvidenceKind.CONTRADICT
                    else "unknown"
                )
                # No conclusive card mentions this technique any more: whatever
                # the belief recorded was derived from evidence that is gone.
                if tech not in totals:
                    confidence, effect = _NEUTRAL, "unknown"

                prior = next((r for r in rows if r.get("id") == canonical), None)
                same = (
                    prior is not None
                    and abs(float(prior.get("confidence") or 0) - confidence) < 1e-9
                    and str(prior.get("effect") or "") == effect
                )
                duplicates = [
                    r
                    for r in rows
                    if r.get("id") != canonical
                    and not _is_evidence_derived(r)
                    # Already merged on a previous run. Without this the rebuild
                    # re-retires the same rows forever and is not idempotent.
                    and not _metadata(r).get("superseded_by")
                ]
                if same and not duplicates:
                    continue

                meta = _metadata(prior) if prior else {}
                meta[_SOURCE_KEY] = _SOURCE_EVIDENCE
                if tech in last_card:
                    meta["last_evidence_card_id"] = last_card[tech]
                if duplicates:
                    # Keep what the literature rows knew, without letting their
                    # citation-count confidence masquerade as a measurement.
                    meta["merged_belief_ids"] = sorted(
                        {*meta.get("merged_belief_ids", []), *(str(d["id"]) for d in duplicates)}
                    )
                    meta["literature_confidence"] = max(
                        float(d.get("confidence") or 0) for d in duplicates
                    )

                store.upsert_belief(
                    belief_id=canonical,
                    technique=tech,
                    status=_belief_status(confidence, kind),
                    effect=effect,
                    confidence=confidence,
                    metadata=meta,
                )
                for dup in duplicates:
                    _retire_duplicate(store, str(dup["id"]), canonical)
                changed.append(canonical)
                logger.info(
                    "Re-derived belief %s: effect=%s confidence=%.2f (from %d card(s))",
                    canonical,
                    effect,
                    confidence,
                    len(cards),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("belief re-derivation failed: %s", exc)
        return changed
    return changed


def _retire_duplicate(store: KnowledgeStore, belief_id: str, canonical: str) -> None:
    """Point a duplicate row at the canonical one instead of deleting it.

    Deleting would lose the literature evidence attached to it. Zeroing its
    confidence stops it being picked up by anything that iterates beliefs and
    keys by technique, which is how it outvoted the measured row.
    """
    try:
        row = store.get_belief(belief_id)
        if row is None:
            return
        meta = _metadata(row)
        meta["superseded_by"] = canonical
        meta["superseded_reason"] = "duplicate belief identity merged into the evidence-derived row"
        store.upsert_belief(
            belief_id=belief_id,
            technique=str(row.get("technique") or ""),
            status="superseded",
            effect="unknown",
            confidence=0.0,
            metadata=meta,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not retire duplicate belief %s: %s", belief_id, exc)
