"""Resolve prior-best parent hypothesis / execution for improve-on-prior plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.hypothesis import (
    BASELINE_HYPOTHESIS_ID,
    HypothesisStore,
)
from labpilot.research_engine.shared.experiments.models import Hypothesis, HypothesisStatus


def resolve_parent_context(
    hypothesis: Hypothesis,
    *,
    knowledge_dir: Path,
    competition: str,
) -> dict[str, Any]:
    """Return parent_* metadata for plan.metadata / PlanningContext."""
    store = HypothesisStore(knowledge_dir, competition)
    parent_id = hypothesis.parent_hypothesis_id
    if not parent_id:
        for tag in hypothesis.tags:
            if str(tag).lower().startswith("fork:"):
                parent_id = str(tag).split(":", 1)[1].strip()
                break
    parent = store.get(parent_id) if parent_id else None
    if parent is None:
        parent = _best_succeeded(store)

    meta: dict[str, Any] = {
        "parent_hypothesis_id": parent.id if parent else None,
        "parent_execution_id": None,
        "parent_metrics": {},
        "parent_actual_outcome": parent.actual_outcome if parent else None,
        "parent_technique": parent.technique if parent else None,
        "parent_technique_stack": list(parent.technique_stack) if parent else [],
        "technique": hypothesis.technique,
        "technique_stack": list(hypothesis.technique_stack),
        "combo_techniques": list(hypothesis.combo_techniques),
        "change_category": _change_category(hypothesis),
        "evidence": [e.model_dump(mode="json") for e in hypothesis.evidence],
    }
    if parent is None:
        return meta

    # Best-effort: find execution outcome artifact mentioning parent hyp.
    exec_meta = _find_parent_execution(knowledge_dir, competition, parent.id)
    meta.update(exec_meta)
    return meta


def _best_succeeded(store: HypothesisStore) -> Hypothesis | None:
    confirmed = [
        h
        for h in store.list()
        if h.status == HypothesisStatus.CONFIRMED and h.id != BASELINE_HYPOTHESIS_ID
    ]
    if confirmed:
        return max(
            confirmed,
            key=lambda h: (
                float(h.public_score) if h.public_score is not None else -1.0,
                float(h.expected_impact or 0.0),
            ),
        )
    baseline = store.get(BASELINE_HYPOTHESIS_ID)
    return baseline


def _change_category(hypothesis: Hypothesis) -> str:
    hay = " ".join(
        [
            hypothesis.technique or "",
            *hypothesis.combo_techniques,
            *hypothesis.tags,
            hypothesis.observation,
            hypothesis.prediction,
        ]
    ).lower()
    if "combination" in hay or len(hypothesis.combo_techniques) >= 2:
        # Prefer FE when a combo member is feature work (template selection).
        if any(
            tok in hay
            for tok in (
                "feature",
                "encod",
                "tfidf",
                "binning",
                "aggregat",
                "feature_engineering",
            )
        ):
            return "feature_engineering"
        if any(
            tok in hay
            for tok in ("augment", "mixup", "cutmix", "specaugment", "cutout")
        ):
            return "augmentation"
        if any(
            tok in hay
            for tok in (
                "model",
                "xgboost",
                "lightgbm",
                "catboost",
                "resnet",
                "transformer",
                "architecture",
            )
        ):
            return "model"
        return "combination"
    if any(
        tok in hay
        for tok in (
            "feature",
            "encod",
            "tfidf",
            "binning",
            "aggregat",
            "feature_engineering",
        )
    ):
        return "feature_engineering"
    if any(
        tok in hay
        for tok in ("augment", "mixup", "cutmix", "specaugment", "cutout")
    ):
        return "augmentation"
    if any(
        tok in hay
        for tok in (
            "model",
            "xgboost",
            "lightgbm",
            "catboost",
            "resnet",
            "transformer",
            "architecture",
        )
    ):
        return "model"
    return "other"


def _find_parent_execution(
    knowledge_dir: Path,
    competition: str,
    parent_hypothesis_id: str,
) -> dict[str, Any]:
    try:
        from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
        from labpilot.research_engine.intelligence.models import ResearchArtifactType

        with KnowledgeStore(knowledge_dir, competition) as store:
            for art in store.list_artifacts(type=ResearchArtifactType.EXPERIMENT):
                meta = art.metadata or {}
                hyp_id = str(meta.get("hypothesis_id") or "")
                if hyp_id != parent_hypothesis_id:
                    continue
                return {
                    "parent_execution_id": str(
                        meta.get("execution_id") or art.id.replace("exp:execution:", "")
                    ),
                    "parent_metrics": dict(meta.get("metrics") or {}),
                }
    except Exception:
        pass
    return {"parent_execution_id": None, "parent_metrics": {}}
