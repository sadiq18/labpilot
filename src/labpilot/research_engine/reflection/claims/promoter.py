"""ClaimPromoter — promote strong beliefs to research_claims."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.reflection.store import ReflectionStore

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 0.7
_MIN_STRONG_EVIDENCE = 1

#: Below this, a technique's measured contribution is treated as no effect.
#: Matches `evidence/builder.py::_claim_updates_from_attribution`, which already
#: refuses to mint a claim for a zero-credit technique. That rule existed in one
#: place and was missing here, so a belief could become a claim on *confidence
#: alone* — measured 2026-08-07 on rogii, where "vit improves the primary
#: metric" was promoted to `supported` at 0.62 confidence while both vit runs
#: scored 194.80084243002463, byte-identical to the untouched baseline.
_NO_EFFECT_EPSILON = 1e-9


class ClaimPromoter:
    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.competition = competition
        self._reflection = ReflectionStore(Path(knowledge_dir), competition)
        self._knowledge = KnowledgeStore(Path(knowledge_dir), competition)
        self._evidence = EvidenceCardStore(Path(knowledge_dir), competition)

    def close(self) -> None:
        self._reflection.close()
        self._knowledge.close()

    def measured_effect(self, technique: str) -> tuple[int, float]:
        """``(observations, net credit)`` for ``technique`` across evidence cards.

        Evidence cards are the only record of what a technique *did*; beliefs
        record what the system currently thinks. Promotion must consult the
        former, or confidence becomes self-reinforcing — a belief raises its own
        confidence, crosses the threshold, and becomes a claim that nothing
        measured ever supported.
        """
        label = technique.strip().lower()
        if not label:
            return 0, 0.0
        observations = 0
        net = 0.0
        try:
            cards = self._evidence.list()
        except Exception:  # noqa: BLE001 — absent store means "nothing measured"
            return 0, 0.0
        for card in cards:
            for name, credit in (card.technique_attribution or {}).items():
                if str(name).strip().lower() != label:
                    continue
                observations += 1
                try:
                    net += float(credit)
                except (TypeError, ValueError):
                    continue
        return observations, net

    def effect_is_measured(self, technique: str) -> tuple[bool, str]:
        """Whether any run actually attributed a change to ``technique``.

        Deliberately direction-agnostic. Credit sign depends on whether the
        metric is minimised or maximised — `SWA` scored **-3.83** for a genuine
        improvement on MSE — so inferring "positive" from the sign here would be
        wrong half the time. Refusing only on *no effect at all* is the check
        that is correct without knowing the metric's direction, and it is
        exactly the case that produced the false vit claim.
        """
        observations, net = self.measured_effect(technique)
        if observations == 0:
            return False, f"no evidence card attributes any result to {technique!r}"
        if abs(net) < _NO_EFFECT_EPSILON:
            return False, (
                f"{observations} run(s) attributed to {technique!r} produced no "
                "measurable change; an effect claim would be a false finding"
            )
        return True, ""

    def promote_from_belief(
        self,
        belief: dict[str, Any],
        *,
        evidence_id: str | None = None,
        contradicting_evidence_id: str | None = None,
        needs_review: bool = False,
    ) -> dict[str, Any] | None:
        confidence = float(belief.get("confidence") or 0)
        if confidence < _MIN_CONFIDENCE and not contradicting_evidence_id:
            return None

        strong = [
            e
            for e in self._reflection.list_evidence()
            if e.get("strength") == "strong"
        ]
        if (
            len(strong) < _MIN_STRONG_EVIDENCE
            and not contradicting_evidence_id
            and confidence < 0.85
        ):
            return None

        technique = str(belief.get("technique") or "")
        effect = str(belief.get("effect") or "")

        # A belief asserting an effect must be backed by a measured one. Without
        # this, confidence alone promotes — and confidence is produced by the
        # same loop that would consume the claim.
        if effect and effect.lower() not in {"unknown", ""} and not contradicting_evidence_id:
            measured, why = self.effect_is_measured(technique)
            if not measured:
                logger.info("Not promoting %r: %s", technique, why)
                return None

        statement = f"{technique} appears to be {effect} on {self.competition}"
        status = "candidate"
        contradictions: list[str] = []
        if contradicting_evidence_id:
            status = "contested"
            contradictions.append(contradicting_evidence_id)

        meta: dict[str, Any] = {}
        if needs_review:
            meta["needs_review"] = True
        claim = self._reflection.create_claim(
            statement,
            status=status,
            confidence=confidence,
            technique=technique,
            effect=effect,
            promoted_from=belief.get("id"),
            contradictions=contradictions,
            metadata=meta or None,
        )
        link_id = evidence_id or (strong[0]["id"] if strong else None)
        if link_id:
            relation = "contradicts" if contradicting_evidence_id == link_id else "supports"
            self._reflection.link_claim_evidence(
                claim["id"], link_id, relation=relation
            )
        return claim

    def revalidate_claims(self) -> list[dict[str, Any]]:
        """Contest existing claims whose effect no measurement supports.

        Runs on every promotion cycle rather than as a one-off migration. A
        stricter rule that only applies to *new* claims leaves the old ones
        steering the campaign for ever — measured on rogii, where 45 `vit`
        claims kept the Conductor proposing vit long after the technique had
        been shown to change nothing.

        Claims are **contested, never deleted**: the record of what the system
        once believed, and why it stopped, is itself research evidence.
        """
        contested: list[dict[str, Any]] = []
        try:
            claims = self._reflection.list_claims()
        except Exception:  # noqa: BLE001
            return []
        for claim in claims:
            technique = str(claim.get("technique") or "")
            effect = str(claim.get("effect") or "")
            status = str(claim.get("status") or "")
            if not technique or status == "contested":
                continue
            if effect.lower() in {"", "unknown"}:
                continue
            measured, why = self.effect_is_measured(technique)
            if measured:
                continue
            try:
                # Reuses the existing upsert rather than adding a second write
                # path: it already forces `contested` and merges metadata, so
                # the two routes cannot drift apart.
                self._reflection.upsert_claim_by_statement(
                    statement=str(claim.get("statement") or ""),
                    technique=technique,
                    confidence=float(claim.get("confidence") or 0.0),
                    status="contested",
                    effect=effect,
                    metadata={
                        "contested_by": "claim_revalidation",
                        "contested_reason": why,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — self-healing must not break a run
                logger.warning("Could not contest claim %s: %s", claim.get("id"), exc)
                continue
            logger.info("Contested claim %s (%s): %s", claim.get("id"), technique, why)
            contested.append(claim)
        return contested

    def promote_eligible(
        self,
        *,
        evidence_id: str | None = None,
        needs_review: bool = False,
    ) -> list[dict[str, Any]]:
        # Correct what is already recorded before adding to it.
        self.revalidate_claims()
        created: list[dict[str, Any]] = []
        for belief in self._knowledge.list_beliefs():
            if float(belief.get("confidence") or 0) < _MIN_CONFIDENCE:
                continue
            if belief.get("status") == "rejected":
                continue
            claim = self.promote_from_belief(
                belief,
                evidence_id=evidence_id,
                needs_review=needs_review,
            )
            if claim:
                created.append(claim)
        return created
