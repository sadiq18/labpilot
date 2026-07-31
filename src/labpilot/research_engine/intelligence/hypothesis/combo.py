"""Combination-experiment shortlist + candidate conversion (hybrid LLM pick)."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Any

from labpilot.research_engine.intelligence.hypothesis.models import (
    HypothesisCandidate,
    HypothesisCandidateKind,
)
from labpilot.research_engine.intelligence.repositories.models import (
    EffortEstimate,
    ExpectedGain,
)
from labpilot.research_engine.intelligence.retrieval.fetchers import normalize_label
from labpilot.research_engine.shared.experiments.models import (
    HypothesisEvidenceRef,
    HypothesisOrigin,
)

if TYPE_CHECKING:
    from labpilot.research_engine.intelligence.hypothesis.ledger import ExperimentLedger

_FE = frozenset(
    {
        "feature_engineering",
        "feature",
        "encod",
        "tfidf",
        "binning",
        "aggregat",
        "interaction",
        "ratio",
        "arithmetic",
    }
)
_MODEL = frozenset(
    {
        "model",
        "xgboost",
        "lightgbm",
        "catboost",
        "resnet",
        "efficientnet",
        "transformer",
        "backbone",
        "architecture",
    }
)
_AUG = frozenset(
    {"augment", "mixup", "cutmix", "specaugment", "cutout", "mask"}
)
_TRAIN = frozenset(
    {"ema", "swa", "warmup", "scheduler", "regulariz", "dropout", "label_smooth"}
)

SHORTLIST_CAP = 12
DEFAULT_PICK_LIMIT = 3


def technique_category(name: str) -> str:
    lower = (name or "").lower()
    if any(tok in lower for tok in _FE):
        return "feature_engineering"
    if any(tok in lower for tok in _AUG):
        return "augmentation"
    if any(tok in lower for tok in _MODEL):
        return "model"
    if any(tok in lower for tok in _TRAIN):
        return "training_strategy"
    return "other"


def build_combo_shortlist(
    ledger: ExperimentLedger,
    *,
    max_portfolios: int = SHORTLIST_CAP,
) -> list[dict[str, Any]]:
    """Deterministic compatible portfolios of size 2 (and 3 when enough untried)."""
    names = _eligible_techniques(ledger)
    if len(names) < 2:
        return []

    avoid = {
        frozenset({normalize_label(a), normalize_label(b)})
        for a, b in ledger.avoid_pairs
    }
    portfolios: list[dict[str, Any]] = []

    for a, b in combinations(names, 2):
        labels = frozenset({normalize_label(a), normalize_label(b)})
        if labels in avoid:
            continue
        cat_a, cat_b = technique_category(a), technique_category(b)
        diversity = 1.0 if cat_a != cat_b else 0.35
        portfolios.append(
            {
                "id": f"pair:{normalize_label(a)}+{normalize_label(b)}",
                "techniques": [a, b],
                "categories": [cat_a, cat_b],
                "size": 2,
                "diversity_score": diversity,
            }
        )

    if len(names) >= 6:
        for a, b, c in combinations(names[:8], 3):
            labels = {
                frozenset({normalize_label(a), normalize_label(b)}),
                frozenset({normalize_label(a), normalize_label(c)}),
                frozenset({normalize_label(b), normalize_label(c)}),
            }
            if labels & avoid:
                continue
            cats = [technique_category(a), technique_category(b), technique_category(c)]
            diversity = len(set(cats)) / 3.0
            portfolios.append(
                {
                    "id": (
                        f"triple:{normalize_label(a)}+"
                        f"{normalize_label(b)}+{normalize_label(c)}"
                    ),
                    "techniques": [a, b, c],
                    "categories": cats,
                    "size": 3,
                    "diversity_score": diversity,
                }
            )

    portfolios.sort(
        key=lambda p: (-float(p["diversity_score"]), -int(p["size"]), p["id"])
    )
    return portfolios[: max(0, max_portfolios)]


def rule_engine_pick_combos(
    shortlist: list[dict[str, Any]],
    *,
    limit: int = DEFAULT_PICK_LIMIT,
) -> list[dict[str, Any]]:
    """Offline pick: highest category diversity, then size 2 before 3."""
    ranked = sorted(
        shortlist,
        key=lambda p: (-float(p.get("diversity_score") or 0), int(p.get("size") or 2), p["id"]),
    )
    picks: list[dict[str, Any]] = []
    used: set[str] = set()
    for portfolio in ranked:
        techs = [str(t) for t in portfolio.get("techniques") or []]
        key = "+".join(normalize_label(t) for t in techs)
        if key in used:
            continue
        used.add(key)
        picks.append(
            {
                "techniques": techs,
                "rationale": (
                    "Rule-engine combo: complementary categories "
                    f"{portfolio.get('categories')}"
                ),
                "confidence": min(0.85, 0.55 + 0.15 * float(portfolio.get("diversity_score") or 0)),
                "expected_impact": 0.02 if float(portfolio.get("diversity_score") or 0) >= 1 else 0.012,
            }
        )
        if len(picks) >= limit:
            break
    return picks


def filter_picks_to_shortlist(
    picks: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop LLM inventions; keep only techniques that appear in the shortlist."""
    allowed: set[str] = set()
    shortlist_keys: set[str] = set()
    for portfolio in shortlist:
        techs = [str(t) for t in portfolio.get("techniques") or []]
        for t in techs:
            allowed.add(normalize_label(t))
        shortlist_keys.add("+".join(sorted(normalize_label(t) for t in techs)))

    cleaned: list[dict[str, Any]] = []
    for pick in picks:
        techs = [str(t).strip() for t in (pick.get("techniques") or []) if str(t).strip()]
        techs = [t for t in techs if normalize_label(t) in allowed]
        if len(techs) < 2:
            continue
        key = "+".join(sorted(normalize_label(t) for t in techs))
        # Prefer exact shortlist membership; allow subset of a shortlisted portfolio.
        if key not in shortlist_keys and not _is_subset_of_shortlist(techs, shortlist):
            continue
        cleaned.append(
            {
                "techniques": techs[:3],
                "rationale": str(pick.get("rationale") or ""),
                "confidence": float(pick.get("confidence") or 0.6),
                "expected_impact": float(pick.get("expected_impact") or 0.015),
            }
        )
    return cleaned


def picks_to_candidates(
    picks: list[dict[str, Any]],
    ledger: ExperimentLedger,
) -> list[HypothesisCandidate]:
    """Convert chosen portfolios into HypothesisCandidate rows."""
    parent_id = ledger.winning_hypothesis_id
    stack = list(ledger.winning_stack)
    out: list[HypothesisCandidate] = []
    for pick in picks:
        techs = [str(t) for t in pick.get("techniques") or []]
        if len(techs) < 2:
            continue
        joined = "+".join(techs)
        label_key = "+".join(normalize_label(t) for t in techs)
        new_stack = list(stack)
        for t in techs:
            if t not in new_stack:
                new_stack.append(t)
        rationale = str(pick.get("rationale") or "Complementary technique merge.")
        conf = min(0.95, max(0.4, float(pick.get("confidence") or 0.6)))
        if parent_id:
            conf = min(0.95, max(conf, ledger.winning_confidence + 0.1))
        impact_val = float(pick.get("expected_impact") or 0.015)
        impact = (
            ExpectedGain.HIGH
            if impact_val >= 0.02
            else ExpectedGain.MEDIUM if impact_val >= 0.01 else ExpectedGain.LOW
        )
        evidence = [
            HypothesisEvidenceRef(
                kind=HypothesisOrigin.EXPERIMENT if parent_id else HypothesisOrigin.MIXED,
                ref=parent_id or "ledger",
                note="combination portfolio",
            )
        ]
        title = (
            f"Combine {' + '.join(techs)} on {parent_id}"
            if parent_id
            else f"Combine {' + '.join(techs)}"
        )
        out.append(
            HypothesisCandidate(
                key=f"combination:{label_key}",
                kind=HypothesisCandidateKind.COMBINATION,
                title=title,
                observation=(
                    f"Unused complementary techniques {techs} can be tested together "
                    f"to reduce sequential experiments. {rationale}"
                ),
                reason=(
                    f"Combination experiment (LLM/rule pick): {rationale} "
                    f"(techniques {joined})"
                ),
                prediction=(
                    f"Applying {joined} together on the prior pipeline will improve "
                    "the primary metric more efficiently than sequential singles."
                ),
                technique=joined,
                expected_impact=impact,
                confidence=conf,
                implementation_effort=EffortEstimate.HOURS_4
                if len(techs) >= 3
                else EffortEstimate.HOURS_1,
                evidence=evidence,
                origins=[HypothesisOrigin.MIXED],
                tags=[*techs, "combination", "stacked", "improvement"]
                + ([f"fork:{parent_id}"] if parent_id else []),
                parent_hypothesis_id=parent_id,
                technique_stack=new_stack,
                score_hint=conf + 0.25 + 0.05 * len(techs),
                metadata={
                    "combo_techniques": techs,
                    "combo_rationale": rationale,
                    "expected_impact_value": impact_val,
                },
            )
        )
    return out


def _eligible_techniques(ledger: ExperimentLedger) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for name in ledger.techniques_untried:
        label = normalize_label(name)
        if not label or label in seen or ledger.is_failed(name):
            continue
        seen.add(label)
        names.append(name)
    for belief in ledger.beliefs_unused:
        tech = str(belief.get("technique") or "").strip()
        label = normalize_label(tech)
        if not tech or not label or label in seen or ledger.is_failed(tech):
            continue
        seen.add(label)
        names.append(tech)
    return names[:12]


def _is_subset_of_shortlist(
    techs: list[str], shortlist: list[dict[str, Any]]
) -> bool:
    want = {normalize_label(t) for t in techs}
    for portfolio in shortlist:
        have = {normalize_label(str(t)) for t in portfolio.get("techniques") or []}
        if want <= have and len(want) >= 2:
            return True
    return False
