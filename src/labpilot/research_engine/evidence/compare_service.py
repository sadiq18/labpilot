"""Resolve control metrics and build Evidence Card for an Engineer COMPARE task."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.evidence.builder import (
    build_evidence_card,
    # Re-exported: this is where the check was defined and where its callers
    # still import it from. It lives in `builder` now because that is where the
    # card is made, and a disqualification the card does not carry is one only
    # this function honours.
    compared_against_itself,
    write_comparison_files,
)
from labpilot.research_engine.evidence.models import EvidenceCard
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.intelligence.graph.writer import write_graph_edges_from_card
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import ResearchArtifactType
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus
from labpilot.research_engine.validation.kaggle import KaggleCvValidator, result_from_metrics

logger = logging.getLogger(__name__)


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _belief_priors(knowledge_dir: Path, competition: str) -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with KnowledgeStore(knowledge_dir, competition) as store:
            for belief in store.list_beliefs():
                tech = str(belief.get("technique") or "").strip()
                if tech:
                    out[tech] = float(belief.get("confidence") or 0.5)
    except Exception as exc:
        logger.debug("belief priors unavailable: %s", exc)
    return out


def resolve_control(
    context: TaskContext,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Return (control_execution_id, control_metrics, control_hypothesis_id)."""
    plan_meta = dict(context.plan.metadata or {})
    control_exec = plan_meta.get("parent_execution_id")
    control_hyp = plan_meta.get("parent_hypothesis_id")
    control_metrics = dict(plan_meta.get("parent_metrics") or {})

    knowledge_dir = context.paths.base_dir
    competition = context.competition

    # Prefer metrics already on plan; else look up experiment artifact.
    if control_exec and not control_metrics:
        control_metrics = _metrics_from_artifact(
            knowledge_dir, competition, str(control_exec)
        )

    if not control_metrics and control_hyp:
        control_exec2, metrics2 = _metrics_for_hypothesis(
            knowledge_dir, competition, str(control_hyp)
        )
        if metrics2:
            control_metrics = metrics2
            control_exec = control_exec or control_exec2

    if not control_metrics:
        # Best confirmed / champion
        store = HypothesisStore(knowledge_dir, competition)
        confirmed = [
            h
            for h in store.list()
            if h.status == HypothesisStatus.CONFIRMED and h.id != "H-BASELINE"
        ]
        if confirmed:
            best = max(
                confirmed,
                key=lambda h: (
                    float(h.public_score) if h.public_score is not None else -1.0,
                    float(h.expected_impact or 0.0),
                ),
            )
            control_hyp = best.id
            control_exec2, metrics2 = _metrics_for_hypothesis(
                knowledge_dir, competition, best.id
            )
            control_metrics = metrics2
            control_exec = control_exec or control_exec2

    return (
        str(control_exec) if control_exec else None,
        control_metrics,
        str(control_hyp) if control_hyp else None,
    )


def _metrics_from_artifact(
    knowledge_dir: Path, competition: str, execution_id: str
) -> dict[str, Any]:
    try:
        with KnowledgeStore(knowledge_dir, competition) as store:
            for art in store.list_artifacts(type=ResearchArtifactType.EXPERIMENT):
                meta = art.metadata or {}
                if str(meta.get("execution_id") or "") == execution_id:
                    metrics = meta.get("metrics")
                    if isinstance(metrics, dict):
                        return dict(metrics)
                if art.id.endswith(execution_id):
                    metrics = meta.get("metrics")
                    if isinstance(metrics, dict):
                        return dict(metrics)
    except Exception:
        pass
    return {}


def _metrics_for_hypothesis(
    knowledge_dir: Path, competition: str, hypothesis_id: str
) -> tuple[str | None, dict[str, Any]]:
    try:
        with KnowledgeStore(knowledge_dir, competition) as store:
            for art in store.list_artifacts(type=ResearchArtifactType.EXPERIMENT):
                meta = art.metadata or {}
                if str(meta.get("hypothesis_id") or "") != hypothesis_id:
                    continue
                metrics = meta.get("metrics")
                if isinstance(metrics, dict) and metrics:
                    return str(meta.get("execution_id") or "") or None, dict(metrics)
    except Exception:
        pass
    return None, {}


def run_compare_and_build_card(context: TaskContext) -> EvidenceCard:
    """COMPARE vs parent control, write comparison.json, persist Evidence Card + graph."""
    root = context.workspace_root
    control_exec, control_metrics, control_hyp = resolve_control(context)
    plan_meta = dict(context.plan.metadata or {})
    knowledge_dir = context.paths.base_dir

    # The production path goes through the validator, so the seam is *used*
    # rather than merely available. Same sources in the same order — the
    # validator loads the same `metrics.json` and resolves direction from the
    # same four places `_resolve_direction` consults — so the card is unchanged.
    #
    # Both sides are described the same way. A comparison between a result and a
    # loose blob is a comparison between two different kinds of thing, and the
    # control is where that asymmetry hid a bug once already.
    result = KaggleCvValidator().validate(context.plan.hypothesis_id, root, context)
    treatment_metrics = result.raw
    control_result = result_from_metrics(control_metrics, maximize=result.maximize)

    card = build_evidence_card(
        knowledge_dir=knowledge_dir,
        competition=context.competition,
        treatment_execution_id=context.execution.id,
        treatment_metrics=treatment_metrics,
        result=result,
        control_result=control_result,
        plan_id=context.plan.id,
        hypothesis_id=context.plan.hypothesis_id or None,
        control_execution_id=control_exec,
        control_metrics=control_metrics,
        control_hypothesis_id=control_hyp,
        plan_metadata=plan_meta,
        belief_priors=_belief_priors(knowledge_dir, context.competition),
        # The run's own competition.json is the nearest record of which way is
        # better; `build_evidence_card` falls back to the knowledge copy and the
        # Analyze profile. Not passing anything here is what inverted every
        # decision on rogii.
        workspace_root=root,
        persist=True,
    )
    write_comparison_files(root, card)
    try:
        write_graph_edges_from_card(
            knowledge_dir=knowledge_dir,
            competition=context.competition,
            card=card,
        )
    except Exception as exc:
        logger.warning("Research graph write failed: %s", exc)
    # The card is still written and still persisted — "this change did nothing"
    # is worth recording. What must not happen is that reading moving a
    # hypothesis to confirmed/rejected, or shifting a belief's confidence, on
    # the strength of a measurement that never varied.
    self_comparison = compared_against_itself(
        str(context.execution.id), control_exec, treatment_metrics, control_metrics
    )
    if self_comparison:
        logger.warning(
            "Not applying evidence for %s: %s",
            context.plan.hypothesis_id or context.plan.id,
            self_comparison,
        )
        return card
    try:
        from labpilot.research_engine.evidence.apply import (
            apply_card_to_beliefs,
            apply_card_to_hypothesis,
        )

        apply_card_to_beliefs(
            knowledge_dir=knowledge_dir,
            competition=context.competition,
            card=card,
        )
        apply_card_to_hypothesis(
            knowledge_dir=knowledge_dir,
            competition=context.competition,
            card=card,
        )
    except Exception as exc:
        logger.warning("Evidence card belief/hyp apply failed: %s", exc)
    return card
