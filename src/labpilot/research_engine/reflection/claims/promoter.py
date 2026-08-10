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
            # `strict`, because this is a measurement: skipping a corrupt card
            # and averaging the survivors changes the number rather than the
            # answer, and the handler below could never fire while the
            # corruption was swallowed upstream. Reported on PR #120.
            cards = self._evidence.list(strict=True)
        except Exception:
            # `(0, 0.0)` is not "unknown", it is *measured, and it was zero* —
            # a claim about evidence, made because the evidence could not be
            # read. The one case that is genuinely nothing-measured returns an
            # empty list rather than raising. M20, 2026-08-09.
            logger.exception(
                "cannot read evidence cards; reporting %r as unmeasured", technique
            )
            return 0, 0.0
        for card in cards:
            if not self._card_compared_something_real(card):
                continue
            for name, credit in (card.technique_attribution or {}).items():
                if str(name).strip().lower() != label:
                    continue
                observations += 1
                try:
                    net += float(credit)
                except (TypeError, ValueError):
                    continue
        return observations, net

    @staticmethod
    def _card_compared_something_real(card: Any) -> bool:
        """Whether this card's gain came from comparing two genuine runs.

        Requires a real verdict (`accepted`/`rejected`) and both scores present.
        `inconclusive` is the builder's own label for a missing control.

        **This does not catch every fabricated score, and must not claim to.**
        An earlier version used ``bool(parent) and bool(treatment)``, which was
        wrong in both directions: it rejected a legitimate `0.0` **treatment**
        (a perfect MSE), and it let `treatment_cv=0.5` stub runs through, since
        `bool(0.5)` is true. The zero-*control* rejection survives on its merits
        and is now scoped to that case alone.

        A stub run that writes a plausible-looking metric is indistinguishable
        from a real one *at this layer*, so the real defence is upstream and now
        exists: `evidence/builder.py::is_placeholder_metrics` refuses to mint a
        card from a run whose metrics say no model was trained, and
        `evidence/repair.py` retires cards already written that way. On rogii
        that moved seven of fifteen cards to `inconclusive`, including the five
        that had been supporting `"hyp:H-010 hurts the primary metric"` — a
        claim about a technique that never existed — with a net credit of
        -971.5.

        This check is the second line: it removes cards that are provably
        uncomparable from their own scores. It does not launder the rest, and a
        workspace whose stub runs left no status marker would still get past it.
        """
        decision = str(getattr(card, "decision", "") or "").lower()
        # Only a card that reached a verdict compared two real runs.
        # `inconclusive` is the builder's own label for a missing control.
        if not ("accepted" in decision or "rejected" in decision):
            return False
        observed = getattr(card, "observed", None)
        parent = getattr(observed, "parent_cv", None)
        treatment = getattr(observed, "treatment_cv", None)
        if parent is None or treatment is None:
            return False
        # A control of exactly zero against a non-zero treatment means the
        # "gain" *is* the entire treatment score — there was no baseline to
        # improve on. rogii's EV-001 credits vit +194.80 that way. Narrowed to
        # the control only: an earlier version also rejected `treatment == 0`,
        # which would discard a perfect MSE.
        return not (parent == 0.0 and treatment != 0.0)

    @staticmethod
    def asserts_an_effect(claim: dict[str, Any]) -> bool:
        """Whether this claim says a technique *did something*.

        Checks the statement as well as the ``effect`` column, because the two
        claim writers disagree about where the assertion lives:

        * `promote_from_belief` sets ``effect`` and writes "X appears to be
          <effect> on <competition>";
        * `_claim_updates_from_attribution` leaves ``effect`` **empty** and puts
          the verb in the statement — "X improves the primary metric".

        Keying on the column alone was a real defect: measured 2026-08-07, all
        seven effect-asserting claims on rogii had ``effect=''``, so a guard
        reading only the column could touch 14 of 417 claims and none of the
        false ones. Including the one that started this: *"vit improves the
        primary metric"*, status `supported`.
        """
        from labpilot.research_engine.evidence.builder import CLAIM_HURTS, CLAIM_IMPROVES

        effect = str(claim.get("effect") or "").strip().lower()
        if effect and effect not in {"unknown", "none"}:
            return True
        # Imported from the writer rather than duplicated: a wording change in
        # `_claim_updates_from_attribution` would otherwise silently disable
        # revalidation for every new claim, and nothing would report it.
        statement = str(claim.get("statement") or "").lower()
        return CLAIM_IMPROVES in statement or CLAIM_HURTS in statement

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
            # Two different situations share this branch and the message must
            # not overstate either: every run measured zero, or runs measured
            # opposing effects that cancel. Both mean the evidence does not
            # support an effect claim; only the first means "changed nothing".
            detail = (
                "no run measured any change"
                if observations == 1
                else f"net attribution across {observations} run(s) is ~0"
            )
            return False, (
                f"{detail} for {technique!r}; the evidence does not support an "
                "effect claim"
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

        # Promotion requires a measured effect. Full stop.
        #
        # This used to be gated on `effect` being set and not "unknown", which
        # made it skippable by exactly the beliefs that needed it most.
        # `KnowledgeHub._persist_belief` writes confidence from *literature
        # mentions* (0.2 + 0.15 per citation, capped at 0.95) and leaves
        # `effect` as "unknown" — so a technique cited five times in retrieved
        # papers arrived at 0.95 confidence and sailed past the guard without
        # ever being measured. That is how "vit improves the primary metric"
        # became a supported claim on a tabular competition.
        #
        # Confidence answers "how often has this been mentioned"; promotion
        # asks "what did it do here". They are different questions and only the
        # second one licenses a claim.
        if not contradicting_evidence_id:
            measured, why = self.effect_is_measured(technique)
            if not measured:
                logger.info("Not promoting %r: %s", technique, why)
                return None

        # Vocabulary filter after measurement: rejected/dormant never become
        # claims. Missing store rows default to candidate — measurement still
        # licenses promotion (novel techniques must not need membership first).
        from labpilot.research_engine.execution.technique.status_constants import (
            CLAIM_BLOCKED_STATUSES,
        )

        vocab_status = self._knowledge.technique_status(technique)
        if vocab_status in CLAIM_BLOCKED_STATUSES:
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
            if not self.asserts_an_effect(claim):
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
