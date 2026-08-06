"""Assemble and decide Evidence Cards from control vs treatment metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.attribution import attribute_techniques
from labpilot.research_engine.evidence.models import (
    ClaimEvidenceKind,
    ClaimUpdate,
    EvidenceCard,
    EvidenceDecision,
    ExpectedOutcomes,
    ObservedOutcomes,
    StabilityOutcome,
)
from labpilot.research_engine.evidence.store import EvidenceCardStore
from labpilot.research_engine.intelligence.competition.direction import resolve_maximize
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import Experiment, Hypothesis
from labpilot.research_engine.shared.labels import is_record_reference

logger = logging.getLogger(__name__)

_NOISE = 0.001


#: Statuses a run writes into metrics.json when it did not actually train.
#: `dry_run_stub` comes from `training/capability.py` (dry run or `train_stub`),
#: `last_resort_scaffold` from the generated fallback script. Both write a
#: plausible-looking number — 0.5 and 0.0 — which is why they fooled every
#: downstream check that only asked whether a score was present.
PLACEHOLDER_STATUSES = frozenset({"dry_run_stub", "last_resort_scaffold"})


def is_placeholder_metrics(metrics: dict[str, Any] | None) -> bool:
    """Whether these metrics came from a run that did not train a model.

    The marker is explicit and has been there the whole time; nothing read it.
    Measured on rogii 2026-08-07, seven of fifteen evidence cards were built
    from placeholder runs — including EV-001, the sole basis of the false claim
    "vit improves the primary metric".
    """
    return str((metrics or {}).get("status") or "").strip().lower() in PLACEHOLDER_STATUSES


def _primary_cv_keyed(metrics: dict[str, Any]) -> tuple[float, str] | None:
    """The primary score *and the key it came from*.

    The key is not bookkeeping. This list mixes metrics that move in opposite
    directions (`cv_accuracy`, `cv_rmse`), so two runs can each yield a number
    from a different key and `treatment - parent` then subtracts an accuracy
    from an RMSE. Measured on rogii 2026-08-07: six cards recorded a "gain" of
    -194.30 by comparing a stub's `cv_accuracy` of 0.5 against a real run's
    `cv_rmse` of 194.80. The caller uses the key to refuse that comparison.
    """
    for key in (
        "cv_score",
        "cv_balanced_accuracy",
        "cv_accuracy",
        "cv_roc_auc",
        "cv_rmse",
        "balanced_accuracy",
        "accuracy",
        "rmse",
        "score",
    ):
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key]), key
    for key, val in metrics.items():
        if key.startswith("cv_") and isinstance(val, (int, float)):
            if key in {"cv_folds", "cv_std", "cv_mean"}:
                continue
            return float(val), key
    return None


def _primary_cv(metrics: dict[str, Any]) -> float | None:
    found = _primary_cv_keyed(metrics)
    return found[0] if found else None


def _float(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if isinstance(metrics.get(key), (int, float)):
            return float(metrics[key])
    return None


def _stability(
    parent_std: float | None, treatment_std: float | None
) -> StabilityOutcome:
    if parent_std is None or treatment_std is None:
        return StabilityOutcome.UNKNOWN
    delta = treatment_std - parent_std
    if abs(delta) < 1e-6:
        return StabilityOutcome.SIMILAR
    # Lower std is better (more stable).
    if delta < -_NOISE:
        return StabilityOutcome.IMPROVED
    if delta > _NOISE:
        return StabilityOutcome.WORSE
    return StabilityOutcome.SIMILAR


def decide_evidence(
    *,
    cv_gain: float | None,
    lb_gain: float | None,
    stability: StabilityOutcome,
    maximize: bool,
    missing_control: bool,
    overfitting: bool = False,
) -> tuple[EvidenceDecision, str]:
    return _decide(
        cv_gain=cv_gain,
        lb_gain=lb_gain,
        stability=stability,
        maximize=maximize,
        missing_control=missing_control,
        overfitting=overfitting,
    )


def _decide(
    *,
    cv_gain: float | None,
    lb_gain: float | None,
    stability: StabilityOutcome,
    maximize: bool,
    missing_control: bool,
    overfitting: bool = False,
) -> tuple[EvidenceDecision, str]:
    if missing_control:
        return EvidenceDecision.INCONCLUSIVE, "missing_control"
    if overfitting:
        return EvidenceDecision.INCONCLUSIVE, "overfit_local_vs_lb"
    signed = cv_gain
    if signed is not None and not maximize:
        signed = -signed
    if stability == StabilityOutcome.WORSE and signed is not None and signed > 0:
        # Gain with severe instability → inconclusive unless LB also clear.
        if lb_gain is None or lb_gain < 0:
            return (
                EvidenceDecision.INCONCLUSIVE,
                "cv_gain_with_worse_stability",
            )
    if lb_gain is not None:
        if lb_gain >= 0 and (signed is None or signed >= -_NOISE):
            return EvidenceDecision.ACCEPTED, "lb_gain_non_negative"
        if lb_gain < -_NOISE:
            return EvidenceDecision.REJECTED, "lb_gain_negative"
    if signed is None:
        return EvidenceDecision.INCONCLUSIVE, "no_cv_delta"
    if abs(signed) < _NOISE:
        return EvidenceDecision.INCONCLUSIVE, "within_noise_epsilon"
    if signed > 0:
        return EvidenceDecision.ACCEPTED, "cv_gain_positive"
    return EvidenceDecision.REJECTED, "cv_gain_negative"


#: The sentence a claim uses to assert an effect. Shared so `ClaimPromoter`
#: can recognise these claims without duplicating the wording — a drift here
#: would silently disable revalidation for every new claim.
CLAIM_IMPROVES = "improves the primary metric"
CLAIM_HURTS = "hurts the primary metric"


def _claim_updates_from_attribution(
    attribution: dict[str, float],
    *,
    decision: EvidenceDecision,
    maximize: bool = True,
) -> list[ClaimUpdate]:
    """Turn per-technique credit into claim updates.

    ``maximize`` is required, not cosmetic. Credit is ``treatment - parent``,
    so on a *minimised* metric a negative credit is an **improvement**. Without
    the flip, every verb is inverted for MSE/RMSE — measured on rogii
    2026-08-07:

    ======================  ========  =====================  ================
    technique               credit    reality (MSE)          claim said
    ======================  ========  =====================  ================
    SWA                     -3.83     improved 194.8->191.0  "hurts"
    vit                     +194.80   worsened               "improves"
    ======================  ========  =====================  ================

    `_decide` already flips on ``maximize``; this function did not, so the
    verdict and the sentence disagreed. "vit improves the primary metric" —
    the claim that started the revalidation work — was wrong for this reason as
    well as for its missing control.
    """
    updates: list[ClaimUpdate] = []
    for tech, credit in attribution.items():
        if abs(credit) < 1e-9:
            continue
        # Signed toward "better": positive means the metric moved the way we
        # want, whichever direction that is. Everything below reads `signed`,
        # never `credit` — orienting only the sentence left the *belief* going
        # the other way, so a card could read "SWA improves the primary metric"
        # while teaching the belief store that SWA is harmful and lowering its
        # confidence. `apply_card_to_beliefs` keys the confidence step and the
        # `effect` off `evidence`, so that half is the half that steers.
        signed = credit if maximize else -credit
        if decision == EvidenceDecision.ACCEPTED and signed > 0:
            kind = ClaimEvidenceKind.SUPPORT
            delta = min(0.12, 0.04 + abs(signed) * 4)
        elif decision == EvidenceDecision.REJECTED or signed < 0:
            kind = ClaimEvidenceKind.CONTRADICT
            delta = -min(0.12, 0.04 + abs(signed) * 4)
        else:
            kind = ClaimEvidenceKind.NEUTRAL
            delta = 0.0
        verb = CLAIM_IMPROVES if signed >= 0 else CLAIM_HURTS
        updates.append(
            ClaimUpdate(
                claim=f"{tech} {verb}",
                evidence=kind,
                confidence_delta=delta,
                technique=tech,
            )
        )
    return updates


#: Kept as a module-local alias so existing call sites read unchanged; the rule
#: itself, and why it has to see the raw label, live in `shared/labels.py`.
_is_record_reference = is_record_reference


def _reusable_for(competition: str, plan_meta: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    change = str(plan_meta.get("change_category") or "").strip()
    if change:
        tags.append(change)
    problem = str(plan_meta.get("problem_type") or "").strip()
    if problem:
        tags.append(problem)
    for tag in plan_meta.get("tags") or []:
        t = str(tag).strip()
        if t and t.lower() not in {"stacked", "combination", "improvement", "technique"}:
            if not _is_record_reference(t):
                tags.append(t)
    # Competition slug tokens as weak modality hints.
    for part in competition.replace("_", "-").split("-"):
        if part in {"audio", "image", "tabular", "nlp", "text", "bird", "cell"}:
            tags.append(part)
    # Dedup preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:12]


def _resolve_direction(
    knowledge_dir: Path, competition: str, workspace_root: Path | None
) -> bool:
    """Metric direction for this competition, or raise saying how to fix it."""
    paths = ResearchPaths(Path(knowledge_dir), competition)
    resolved = resolve_maximize(
        competition=competition,
        workspace_root=workspace_root,
        knowledge_root=paths.root,
        extracted_dir=paths.extracted_dir,
    )
    if resolved is None:
        raise ValueError(
            f"cannot determine whether {competition!r} maximises or minimises its "
            "metric, so the sign of every conclusion on this card would be a "
            "guess. Set metric.direction in the workspace competition.json, or "
            "run analyze to produce the competition profile. Pass maximize= "
            "explicitly only when the caller genuinely knows better."
        )
    return resolved


def build_evidence_card(
    *,
    knowledge_dir: Path,
    competition: str,
    treatment_execution_id: str,
    treatment_metrics: dict[str, Any],
    plan_id: str | None = None,
    hypothesis_id: str | None = None,
    control_execution_id: str | None = None,
    control_metrics: dict[str, Any] | None = None,
    control_hypothesis_id: str | None = None,
    plan_metadata: dict[str, Any] | None = None,
    maximize: bool | None = None,
    workspace_root: Path | None = None,
    lb_gain: float | None = None,
    overfitting: bool = False,
    belief_priors: dict[str, float] | None = None,
    persist: bool = True,
) -> EvidenceCard:
    """Build (and optionally persist) an Evidence Card for one treatment run.

    ``maximize`` is resolved from the competition profile when not given, and a
    direction that cannot be resolved raises. It used to default to ``True``,
    which no caller overrode: on rogii, an MSE competition, that recorded the
    one real improvement the system produced (SWA, 194.80 -> 190.97) as
    ``rejected`` and a regression as ``accepted``. A card is a signed, durable
    conclusion; writing one whose sign is a guess is worse than writing none.
    """
    if maximize is None:
        maximize = _resolve_direction(knowledge_dir, competition, workspace_root)
    plan_meta = dict(plan_metadata or {})
    control_metrics = dict(control_metrics or {})
    missing_control = not control_metrics and not control_execution_id

    parent_found = _primary_cv_keyed(control_metrics) if control_metrics else None
    treatment_found = _primary_cv_keyed(treatment_metrics)
    parent_cv = parent_found[0] if parent_found else None
    treatment_cv = treatment_found[0] if treatment_found else None

    # Two runs scored on different metrics are not comparable, and the
    # subtraction that follows would invent a gain out of a unit mismatch.
    # Treated as a missing control rather than a small gain: we do not know what
    # the control scored *on this metric*, which is precisely `missing_control`.
    mismatched_metric = (
        parent_found is not None
        and treatment_found is not None
        and parent_found[1] != treatment_found[1]
    )
    # A run that never trained has nothing to compare. Refusing here is the
    # upstream fix that `ClaimPromoter._card_compared_something_real` could only
    # describe: at that layer a stub score is indistinguishable from a real one,
    # but at this one the run itself says so.
    placeholder_treatment = is_placeholder_metrics(treatment_metrics)
    placeholder_control = is_placeholder_metrics(control_metrics)
    uncomparable = mismatched_metric or placeholder_treatment or placeholder_control
    cv_gain = None
    if parent_cv is not None and treatment_cv is not None and not uncomparable:
        cv_gain = treatment_cv - parent_cv

    parent_std = _float(control_metrics, "cv_std")
    treatment_std = _float(treatment_metrics, "cv_std")
    stability = _stability(parent_std, treatment_std)

    parent_time = _float(control_metrics, "train_time_s")
    treat_time = _float(treatment_metrics, "train_time_s")
    runtime_frac = None
    if parent_time and parent_time > 0 and treat_time is not None:
        runtime_frac = (treat_time - parent_time) / parent_time

    hyp: Hypothesis | None = None
    if hypothesis_id:
        hyp = HypothesisStore(knowledge_dir, competition).get(hypothesis_id)

    expected_cv = float(hyp.expected_impact) if hyp and hyp.expected_impact else None
    techniques: list[str] = []
    if hyp:
        techniques = list(hyp.combo_techniques) or (
            [hyp.technique] if hyp.technique and "+" not in (hyp.technique or "") else []
        )
        if not techniques and hyp.technique and "+" in hyp.technique:
            techniques = [p.strip() for p in hyp.technique.split("+") if p.strip()]
        if not techniques:
            techniques = [
                t
                for t in hyp.tags
                if t.lower()
                not in {
                    "stacked",
                    "combination",
                    "ablation",
                    "improvement",
                    "technique",
                }
                and not _is_record_reference(t)
            ][:3]

    attribution = attribute_techniques(
        techniques,
        cv_gain=cv_gain or 0.0,
        belief_priors=belief_priors or {},
    )

    decision, reason = _decide(
        cv_gain=cv_gain,
        lb_gain=lb_gain,
        stability=stability,
        maximize=maximize,
        missing_control=missing_control or parent_cv is None or uncomparable,
        overfitting=overfitting,
    )
    if placeholder_treatment or placeholder_control:
        sides = []
        if placeholder_treatment:
            sides.append(f"treatment reported {treatment_metrics.get('status')!r}")
        if placeholder_control:
            sides.append(f"control reported {control_metrics.get('status')!r}")
        reason = f"placeholder_metrics: {', '.join(sides)}; no model was trained"
    elif mismatched_metric:
        reason = (
            f"metric_key_mismatch: control scored {parent_found[1]!r}, "
            f"treatment scored {treatment_found[1]!r}"
        )

    impact_error = None
    if cv_gain is not None and expected_cv is not None:
        impact_error = cv_gain - expected_cv

    card = EvidenceCard(
        competition=competition,
        hypothesis_id=hypothesis_id,
        control_experiment=control_execution_id,
        treatment_experiment=treatment_execution_id,
        control_hypothesis_id=control_hypothesis_id,
        plan_id=plan_id,
        expected=ExpectedOutcomes(cv_gain=expected_cv, runtime=None),
        observed=ObservedOutcomes(
            cv_gain=cv_gain,
            lb_gain=lb_gain,
            runtime=runtime_frac,
            stability=stability,
            parent_cv=parent_cv,
            treatment_cv=treatment_cv,
            parent_cv_std=parent_std,
            treatment_cv_std=treatment_std,
            train_time_s=treat_time,
            inference_time_s=_float(treatment_metrics, "inference_time_s"),
            peak_memory_mb=_float(treatment_metrics, "peak_memory_mb"),
        ),
        technique_attribution=attribution,
        claim_updates=_claim_updates_from_attribution(
            attribution, decision=decision, maximize=maximize
        ),
        decision=decision,
        decision_reason=reason,
        reusable_for=_reusable_for(competition, plan_meta),
        impact_error=impact_error,
        maximize=maximize,
        noise_epsilon=_NOISE,
    )

    if persist:
        card = EvidenceCardStore(knowledge_dir, competition).save(card)
    return card


def write_comparison_files(workspace_root: Path, card: EvidenceCard) -> None:
    """Write comparison.json to workspace root and artifacts/."""
    root = Path(workspace_root)
    payload = card.to_comparison_dict()
    text = json.dumps(payload, indent=2) + "\n"
    (root / "comparison.json").write_text(text, encoding="utf-8")
    arts = root / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)
    (arts / "comparison.json").write_text(text, encoding="utf-8")
    if card.id:
        (arts / f"evidence_card_{card.id}.json").write_text(
            card.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )


def metrics_as_experiment(
    execution_id: str,
    competition: str,
    metrics: dict[str, Any],
    *,
    runtime_seconds: float | None = None,
) -> Experiment:
    """Minimal Experiment view for comparator.compare."""
    from datetime import UTC, datetime

    numeric = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    return Experiment(
        id=execution_id,
        competition=competition,
        status="completed",
        progress="done",
        description=execution_id,
        metrics=numeric,
        runtime_seconds=runtime_seconds
        or (float(metrics["train_time_s"]) if isinstance(metrics.get("train_time_s"), (int, float)) else None),
        created_at=datetime.now(UTC),
    )
