"""Submit + learn — upload execution-scoped CSV and patch knowledge with LB score."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.accessor.kaggle.client import KaggleClient, SubmissionResult
from labpilot.config import KaggleConfig
from labpilot.research_engine.execution.outcome import (
    ExecutionOutcomeSummary,
    LeaderboardOutcome,
    _extract_public_score,
    _primary_metric,
    experiment_artifact_id,
    list_execution_submission_csvs,
    load_execution_outcome,
    package_execution_submission,
    submission_csv_path,
    submission_result_path,
    upsert_experiment_artifact,
    write_outcome_files,
)
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import HypothesisStatus

logger = logging.getLogger(__name__)


class SubmitLearnError(RuntimeError):
    """Fatal submit/learn failure."""


def resolve_submission_csv(
    workspace_root: Path,
    execution_id: str,
    *,
    path: Path | None = None,
) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.is_file():
            raise SubmitLearnError(f"submission file not found: {resolved}")
        return resolved
    scoped = submission_csv_path(workspace_root, execution_id)
    if scoped.is_file():
        return scoped
    available = list_execution_submission_csvs(workspace_root)
    listing = ", ".join(p.name for p in available) or "(none)"
    raise SubmitLearnError(
        f"No {scoped.name} under {Path(workspace_root) / 'artifacts'}. "
        f"Available: {listing}. Pass --path to override."
    )


def _prior_public_scores(
    knowledge_dir: Path,
    competition: str,
    *,
    exclude_execution_id: str,
) -> float | None:
    """Best prior public_score from other experiment artifacts / hyp files."""
    scores: list[float] = []
    with KnowledgeStore(knowledge_dir, competition) as store:
        for art in store.list_artifacts(type="experiment"):
            if art.id == experiment_artifact_id(exclude_execution_id):
                continue
            lb = (art.metadata or {}).get("leaderboard") or {}
            if isinstance(lb.get("public_score"), (int, float)):
                scores.append(float(lb["public_score"]))
    store = HypothesisStore(knowledge_dir, competition)
    for hyp in store.list():
        if hyp.public_score is not None:
            scores.append(float(hyp.public_score))
    if not scores:
        return None
    return max(scores)


def _detect_overfitting(
    *,
    local_score: float | None,
    learning_gain: float | None,
    public_score: float | None,
    prior_public: float | None,
) -> bool:
    if public_score is None:
        return False
    local_strong = (learning_gain is not None and learning_gain > 0.0) or (
        local_score is not None
    )
    if not local_strong:
        return False
    if prior_public is not None and public_score < prior_public:
        return True
    # Local improved but public clearly worse than local (higher-is-better heuristic
    # with a relative gap). For lower-is-better metrics this is imperfect; still a
    # useful signal when local CV looks strong.
    if local_score is not None and learning_gain is not None and learning_gain > 0:
        if public_score + 1e-9 < local_score * 0.98:
            return True
    return False


def _write_submission_result(
    path: Path,
    *,
    execution_id: str,
    competition: str,
    submission_path: Path,
    result: SubmissionResult | None,
    upload_meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "execution_id": execution_id,
        "competition": competition,
        "path": str(submission_path),
        "public_score": result.public_score if result else upload_meta.get("public_score"),
        "status": result.status if result else upload_meta.get("status", "unknown"),
        "message": result.message if result else upload_meta.get("message", ""),
        "submissions_url": (
            result.submissions_url if result else upload_meta.get("submissions_url")
        ),
        "upload": upload_meta,
    }
    if result is not None:
        payload["result"] = result.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    # Convenience latest.
    latest = path.parent / "submission_result.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _apply_submit_knowledge(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    plan_tags: list[str],
    overfitting: bool,
) -> str | None:
    """Update hyp public_score, beliefs, techniques, claims; mint only if worth it."""
    from labpilot.research_engine.execution.outcome import (
        maybe_mint_improvement_hypothesis,
        notify_proposed_hypotheses,
        promote_outcome_claims,
    )

    hyp_store = HypothesisStore(knowledge_dir, competition)
    public = summary.leaderboard.public_score if summary.leaderboard else None
    actual_bits = []
    if summary.hypothesis_outcome.get("actual_outcome"):
        actual_bits.append(str(summary.hypothesis_outcome["actual_outcome"]))
    if public is not None:
        actual_bits.append(f"public_score={public:.6g}")
    if overfitting:
        actual_bits.append(
            "OVERFIT signal: strong local metrics but weaker public LB score"
        )
    actual = "; ".join(actual_bits) if actual_bits else None

    status = None
    if overfitting:
        status = HypothesisStatus.INCONCLUSIVE
    elif (
        public is not None
        and summary.leaderboard
        and summary.leaderboard.delta_vs_prior is not None
    ):
        # Public LB vs prior is the ground-truth confirm signal. Do not require
        # local learning_gain (often missing when comparison.json was absent).
        if summary.leaderboard.delta_vs_prior >= 0:
            status = HypothesisStatus.CONFIRMED
        else:
            status = HypothesisStatus.REJECTED
    elif (
        summary.learning_gain is not None
        and summary.learning_gain > 0
        and public is not None
        and (summary.leaderboard is None or summary.leaderboard.delta_vs_prior is None)
    ):
        # Scored, local gain, but no prior LB to compare — leave inconclusive.
        status = HypothesisStatus.INCONCLUSIVE

    if status is not None:
        summary.hypothesis_outcome["status"] = status.value

    # Patch Evidence Card with LB gain and re-apply beliefs (step, never overwrite).
    try:
        from labpilot.research_engine.evidence.apply import (
            apply_card_to_beliefs,
            apply_card_to_hypothesis,
        )
        from labpilot.research_engine.evidence.builder import build_evidence_card
        from labpilot.research_engine.evidence.store import EvidenceCardStore
        from labpilot.research_engine.intelligence.graph.writer import (
            write_graph_edges_from_card,
        )

        store = EvidenceCardStore(knowledge_dir, competition)
        card = store.get_for_execution(summary.execution_id)
        lb_delta = (
            summary.leaderboard.delta_vs_prior if summary.leaderboard else None
        )
        if card is not None:
            observed = card.observed.model_copy(update={"lb_gain": lb_delta})
            # Rebuild decision with LB signal.
            from labpilot.research_engine.evidence.builder import decide_evidence

            decision, reason = decide_evidence(
                cv_gain=card.observed.cv_gain,
                lb_gain=lb_delta,
                stability=card.observed.stability,
                maximize=card.maximize,
                missing_control=(
                    card.control_experiment is None and card.observed.parent_cv is None
                )
                # A self-comparison has a control id and a parent_cv, so it
                # cleared the test above; with `cv_gain` None and a non-negative
                # leaderboard delta `_decide` then returned `accepted`, and this
                # function applied it to the belief and the hypothesis — the one
                # re-derivation that never consulted the check at all.
                or card.uncomparable_reason is not None,
                overfitting=overfitting,
            )
            card = card.model_copy(
                update={
                    "observed": observed,
                    "decision": decision,
                    "decision_reason": reason,
                }
            )
            card = store.save(card)
            write_graph_edges_from_card(
                knowledge_dir=knowledge_dir,
                competition=competition,
                card=card,
            )
            apply_card_to_beliefs(
                knowledge_dir=knowledge_dir,
                competition=competition,
                card=card,
            )
            apply_card_to_hypothesis(
                knowledge_dir=knowledge_dir,
                competition=competition,
                card=card,
            )
            summary.hypothesis_outcome["evidence_card_id"] = card.id
            summary.hypothesis_outcome["decision"] = card.decision.value
            if status is None:
                status = {
                    "accepted": HypothesisStatus.CONFIRMED,
                    "rejected": HypothesisStatus.REJECTED,
                    "inconclusive": HypothesisStatus.INCONCLUSIVE,
                }.get(card.decision.value)
                if status is not None:
                    summary.hypothesis_outcome["status"] = status.value
    except Exception as exc:
        logger.warning("Evidence card LB patch skipped: %s", exc)

    if summary.hypothesis_id:
        try:
            why = "Public score recorded after submit."
            if overfitting:
                why = "Public LB weaker than local CV — likely overfitting."
            elif status == HypothesisStatus.CONFIRMED:
                delta = summary.leaderboard.delta_vs_prior if summary.leaderboard else None
                why = (
                    f"Public LB beat prior "
                    f"(delta_vs_prior={delta:+.6g})."
                    if delta is not None
                    else "Public LB recorded; confirmed."
                )
            elif status == HypothesisStatus.REJECTED:
                delta = summary.leaderboard.delta_vs_prior if summary.leaderboard else None
                why = (
                    f"Public LB worse than prior "
                    f"(delta_vs_prior={delta:+.6g})."
                    if delta is not None
                    else "Public LB recorded; rejected."
                )
            hyp_store.update_outcome(
                summary.hypothesis_id,
                actual_outcome=actual,
                public_score=public,
                status=status,
                evidence_run_id=summary.execution_id,
                why=why,
            )
        except FileNotFoundError:
            logger.warning("Hypothesis %s missing during submit learn", summary.hypothesis_id)

    notify_note = f"Submission scored for {summary.execution_id}"
    if public is not None:
        notify_note += f" public_score={public:.6g}"
    if overfitting:
        notify_note += " (overfit vs local CV)"
    notify_proposed_hypotheses(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        note=notify_note,
    )

    technique = plan_tags[0] if plan_tags else "baseline"
    belief_id = f"belief:{competition}:{technique.replace(' ', '-')[:48]}"
    if overfitting:
        belief_id = f"belief:{competition}:overfitting-{technique.replace(' ', '-')[:40]}"

    with KnowledgeStore(knowledge_dir, competition) as kstore:
        if overfitting:
            kstore.upsert_belief(
                belief_id=belief_id,
                technique=technique,
                status="validated",
                effect="negative",
                confidence=0.7,
                metadata={
                    "kind": "overfitting",
                    "execution_id": summary.execution_id,
                    "public_score": public,
                    "local_metrics": summary.metrics,
                    "message": (
                        "Local CV looked good but public LB did not — "
                        "prefer stronger validation / regularization."
                    ),
                },
            )
            tid = kstore.merge_technique(
                "overfitting",
                category="risk",
                summary="Local metrics can diverge from public LB",
                known_issues="High train/CV with weak public score",
                confidence=0.7,
                evidence=[experiment_artifact_id(summary.execution_id)],
            )
            kstore.link_artifact_technique(
                experiment_artifact_id(summary.execution_id),
                tid,
                relation="overfits",
                weight=1.0,
            )
        else:
            # Step belief confidence from prior (do not overwrite absolute values).
            existing = kstore.get_belief(belief_id)
            prior = float(existing["confidence"]) if existing else 0.5
            effect = "positive" if (summary.learning_gain or 0) > 0 else "unknown"
            if summary.leaderboard and summary.leaderboard.delta_vs_prior is not None:
                if summary.leaderboard.delta_vs_prior >= 0:
                    effect = "positive"
                    prior = min(0.99, prior + 0.06)
                else:
                    effect = "negative"
                    prior = max(0.05, prior - 0.08)
            elif effect == "positive":
                prior = min(0.99, prior + 0.04)
            kstore.upsert_belief(
                belief_id=belief_id,
                technique=technique,
                status="suggested" if prior < 0.65 else "validated",
                effect=effect,
                confidence=prior,
                metadata={
                    "execution_id": summary.execution_id,
                    "public_score": public,
                    "stepped_from_submit": True,
                },
            )
            tid = kstore.merge_technique(
                technique,
                category="experiment",
                summary=f"Used in {summary.execution_id}",
                evidence=[experiment_artifact_id(summary.execution_id)],
            )
            relation = "supports" if effect == "positive" else "mentions"
            kstore.link_artifact_technique(
                experiment_artifact_id(summary.execution_id),
                tid,
                relation=relation,
            )

    promote_outcome_claims(knowledge_dir=knowledge_dir, competition=competition)

    # Only mint when overfit (actionable generalization fork) or other worth-trying signal.
    return maybe_mint_improvement_hypothesis(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        reflection={
            "assessment": {
                "summary": actual,
                "recommendation": (
                    "Add regularization / tighten validation so public LB tracks local CV"
                    if overfitting
                    else ""
                ),
                "confidence": 0.75 if overfitting else 0.5,
                "likely_cause": "overfitting" if overfitting else "",
            }
        },
        overfitting=overfitting,
    )


def submit_and_learn(
    *,
    knowledge_dir: Path,
    competition: str,
    execution_id: str,
    workspace_root: Path | None = None,
    submission_path: Path | None = None,
    message: str | None = None,
    kaggle_config: KaggleConfig | None = None,
    dry_run: bool = False,
    client: KaggleClient | None = None,
) -> ExecutionOutcomeSummary:
    """Upload ``submission_<E-id>.csv``, patch outcome artifact, mutate knowledge."""
    exec_store = ExecutionStore(knowledge_dir, competition)
    try:
        execution = exec_store.get_execution(execution_id)
    finally:
        exec_store.close()
    if execution is None:
        raise SubmitLearnError(f"unknown execution_id: {execution_id}")

    root = Path(workspace_root or execution.workspace_path)
    if not root.is_dir():
        raise SubmitLearnError(f"workspace not found: {root}")

    csv_path = resolve_submission_csv(root, execution_id, path=submission_path)
    # Ensure packaging exists even if user only had predictions.
    if submission_path is None and not submission_csv_path(root, execution_id).is_file():
        package_execution_submission(root, execution_id)
        csv_path = submission_csv_path(root, execution_id)

    plan_store = PlanStore(knowledge_dir, competition)
    try:
        plan = plan_store.get_plan(execution.plan_id)
    finally:
        plan_store.close()
    plan_tags = list((plan.metadata.get("tags") if plan else None) or [])
    if plan and plan.metadata.get("plan_kind"):
        plan_tags.append(str(plan.metadata["plan_kind"]))
    if not plan_tags:
        plan_tags = ["baseline"]

    paths = ResearchPaths(knowledge_dir, competition)
    summary = load_execution_outcome(root, paths=paths, execution_id=execution_id)
    if summary is None and plan is not None:
        from labpilot.research_engine.execution.outcome import build_execution_outcome

        summary = build_execution_outcome(
            knowledge_dir=knowledge_dir,
            competition=competition,
            execution=execution,
            plan=plan,
            workspace_root=root,
            reflection={},
        )
    if summary is None:
        raise SubmitLearnError(
            f"No execution outcome for {execution_id}; re-run the plan successfully first."
        )

    upload_meta: dict[str, Any] = {"uploaded": False, "dry_run": dry_run}
    result: SubmissionResult | None = None

    if dry_run:
        upload_meta["reason"] = "dry-run; no upload"
        upload_meta["status"] = "dry_run"
    else:
        if kaggle_config is None:
            raise SubmitLearnError("kaggle_config required for live submit")
        # Quota preflight
        kclient = client or KaggleClient(kaggle_config)
        try:
            today = kclient.count_todays_submissions(competition)
            meta = kclient.fetch_competition_metadata(competition)
            max_daily = getattr(meta, "max_daily_submissions", None) if meta else None
            if max_daily is not None and today >= int(max_daily):
                raise SubmitLearnError(
                    f"Daily submission quota reached ({today}/{max_daily}) for {competition}"
                )
        except SubmitLearnError:
            raise
        except Exception as exc:
            logger.warning("Quota preflight skipped: %s", exc)

        try:
            result = kclient.upload_submission(
                competition,
                csv_path,
                message=message or f"labpilot {execution_id}",
            )
            upload_meta = {
                "uploaded": True,
                "dry_run": False,
                "public_score": result.public_score,
                "status": result.status,
                "message": result.message,
                "submissions_url": result.submissions_url,
                "result": result.model_dump(mode="json"),
            }
        except Exception as exc:
            upload_meta = {
                "uploaded": False,
                "dry_run": False,
                "error": str(exc),
                "status": "error",
            }
            _write_submission_result(
                submission_result_path(root, execution_id),
                execution_id=execution_id,
                competition=competition,
                submission_path=csv_path,
                result=None,
                upload_meta=upload_meta,
            )
            raise SubmitLearnError(f"upload failed: {exc}") from exc

    _write_submission_result(
        submission_result_path(root, execution_id),
        execution_id=execution_id,
        competition=competition,
        submission_path=csv_path,
        result=result,
        upload_meta=upload_meta,
    )

    public_score = result.public_score if result else _extract_public_score(upload_meta)
    prior = _prior_public_scores(
        knowledge_dir, competition, exclude_execution_id=execution_id
    )
    local = _primary_metric(summary.metrics)
    delta_prior = (
        (public_score - prior) if public_score is not None and prior is not None else None
    )
    delta_local = (
        (public_score - local) if public_score is not None and local is not None else None
    )
    overfitting = False
    if not dry_run:
        overfitting = _detect_overfitting(
            local_score=local,
            learning_gain=summary.learning_gain,
            public_score=public_score,
            prior_public=prior,
        )

    summary.leaderboard = LeaderboardOutcome(
        public_score=public_score,
        prior_public_score=prior,
        delta_vs_prior=delta_prior,
        delta_vs_local=delta_local,
        scored_at=datetime.now(UTC).isoformat(),
        submissions_url=(
            result.submissions_url if result else upload_meta.get("submissions_url")
        ),
        overfitting=overfitting if not dry_run else None,
    )
    summary.submission_path = str(csv_path)
    summary.hypothesis_outcome["public_score"] = public_score
    if summary.hypothesis_outcome.get("actual_outcome") and public_score is not None:
        summary.hypothesis_outcome["actual_outcome"] = (
            f"{summary.hypothesis_outcome['actual_outcome']}; "
            f"public_score={public_score:.6g}"
            + (" [overfit]" if overfitting else "")
        )
    summary.updated_at = datetime.now(UTC).isoformat()

    if not dry_run and public_score is not None:
        follow_id = _apply_submit_knowledge(
            knowledge_dir=knowledge_dir,
            competition=competition,
            summary=summary,
            plan_tags=plan_tags,
            overfitting=overfitting,
        )
        summary.follow_up_hypothesis_id = follow_id

    write_outcome_files(summary, workspace_root=root, paths=paths)
    upsert_experiment_artifact(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        techniques=plan_tags,
    )
    return summary
