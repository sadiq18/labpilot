"""Rank hypotheses by what measurement says, not by what generation guessed.

`_next_hypothesis_id` sorted by ``(confidence, id)`` — so it *was* ranking by
confidence, which is not the flaw it first appears. The flaw is that
``confidence`` is written once at creation and **never updated by evidence**:

* `HypothesisStore.create(confidence=...)` sets it;
* nothing anywhere writes it again.

It is a prior with no posterior. Measured on rogii, `hyp:H-010` sat at 0.99
through the runs that disproved it, and a hypothesis whose technique had been
measured as *harmful* still outranked one nobody had tried.

### The posterior is derived, never stored

Beliefs already hold what measurement concluded, per technique, re-derived from
the current evidence cards by `rederive_beliefs_from_cards`:

    SWA               | positive | 0.62
    rolling_features  | negative | 0.38

So the score is computed at selection time from the prior *and* those beliefs,
rather than written back onto the hypothesis. Storing it would create a fourth
derived value that drifts from its source — the failure this project has now
fixed for plan projections, evidence-card dumps and skill overlays. Deriving it
means a card repaired tomorrow changes the ranking tomorrow, with nothing to
migrate.

### The prior still counts

A measured technique moves the score; it does not replace it. An untested
hypothesis with a strong prior should still outrank a weak one, and evidence
about `SWA` says nothing about a hypothesis proposing something else.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: How far measurement may move a prior, in either direction.
#:
#: Chosen by arithmetic, not by feel. Evidence must be able to overturn a
#: confident prior — otherwise "disproved outranks untried" survives, which is
#: the defect this module exists to remove — while weak evidence must not.
#:
#: At 0.4 the swing is bounded by ±0.4, so a 0.9 prior floors at 0.5 and can
#: never sink below an untried 0.5 *however strong* the evidence. That is a
#: ceiling on measurement, and it is the wrong way round.
#:
#: At 0.6, against an untried 0.5:
#:
#: | prior | belief | score | outranked by untried? |
#: |---|---|---|---|
#: | 0.9 | −0.9 (strong negative) | 0.36 | **yes** |
#: | 0.9 | −0.38 (weak negative)  | 0.67 | no |
#: | 0.5 | +0.62 (positive)       | 0.87 | promoted |
#:
#: Strong measurement wins; a tentative belief does not overturn everything
#: generation knew. `rolling_features | negative | 0.38` on rogii is exactly
#: the weak case, and it *should* leave a confident prior standing.
_EVIDENCE_WEIGHT = 0.6

#: Belief effects and which direction they push.
_EFFECT_SIGN: dict[str, float] = {"positive": 1.0, "negative": -1.0}


def _technique_labels(hypothesis: object) -> set[str]:
    """Every technique name this hypothesis is about, normalised."""
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    names: list[str] = []
    for attr in ("technique",):
        value = getattr(hypothesis, attr, None)
        if value:
            names.append(str(value))
    for attr in ("technique_stack", "combo_techniques"):
        names.extend(str(v) for v in (getattr(hypothesis, attr, None) or []))
    return {key for key in (normalize_label(n) for n in names) if key}


def measured_effects(knowledge_dir: Path, competition: str) -> dict[str, float]:
    """``{normalised technique: signed strength}`` from current beliefs.

    Strength is the belief's own confidence, signed by its effect, so a
    confident negative outweighs a tentative one. Techniques with `unknown`
    effect contribute nothing rather than zero-weight noise.
    """
    from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
    from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label

    effects: dict[str, float] = {}
    try:
        with KnowledgeStore(Path(knowledge_dir), competition) as store:
            rows = store.list_beliefs()
    except Exception as exc:  # noqa: BLE001 — no beliefs means no posterior
        logger.debug("no beliefs available for ranking: %s", exc)
        return {}

    for row in rows:
        technique = normalize_label(str(row.get("technique") or ""))
        sign = _EFFECT_SIGN.get(str(row.get("effect") or "").strip().lower())
        if not technique or sign is None:
            continue
        try:
            strength = float(row.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        # Keep the strongest claim per technique. Two beliefs about one
        # technique disagreeing is a repair problem, not a ranking one.
        signed = sign * strength
        if abs(signed) > abs(effects.get(technique, 0.0)):
            effects[technique] = signed
    return effects


def posterior_score(hypothesis: object, effects: dict[str, float]) -> float:
    """The prior, moved by whatever measurement says about its techniques.

    Returns the prior unchanged when nothing has been measured about this
    hypothesis — an untested idea is neither promoted nor punished, which is
    what keeps a fresh pool ordered by generation's own judgement.
    """
    try:
        prior = float(getattr(hypothesis, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        prior = 0.0
    if not effects:
        return prior

    matched = [effects[label] for label in _technique_labels(hypothesis) if label in effects]
    if not matched:
        return prior
    # Mean, not sum: a hypothesis naming three techniques is not three times as
    # well evidenced, and summing would rank combos above singles on count.
    adjustment = sum(matched) / len(matched)
    return max(0.0, min(1.0, prior + _EVIDENCE_WEIGHT * adjustment))


def rank_hypotheses(
    hypotheses: list,
    knowledge_dir: Path,
    competition: str,
    *,
    effects: dict[str, float] | None = None,
) -> list:
    """Best first, by posterior score then id for a stable order.

    Ties break on id so two runs over the same store choose the same
    hypothesis — a selector that varies run to run makes a campaign
    irreproducible for reasons that have nothing to do with research.

    ``effects`` is injectable so the ordering can be exercised against known
    measurements. Production passes nothing and they are loaded from beliefs;
    without the seam the only way to test "disproved ranks below untried" is to
    build a store, and a test that hard to write is a test nobody writes.
    """
    if effects is None:
        effects = measured_effects(Path(knowledge_dir), competition)
    return sorted(
        hypotheses,
        key=lambda h: (-posterior_score(h, effects), str(getattr(h, "id", ""))),
    )
