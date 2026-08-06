"""Execution outcome summary — local learning card after a successful plan run."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore
from labpilot.research_engine.intelligence.models import ResearchArtifact, ResearchArtifactType
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.planner.schemas.models import ResearchPlan
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import (
    HypothesisCreatedBy,
    HypothesisGenerator,
    HypothesisOrigin,
    HypothesisStatus,
)

logger = logging.getLogger(__name__)


def experiment_artifact_id(execution_id: str) -> str:
    return f"exp:execution:{execution_id}"


def submission_csv_name(execution_id: str) -> str:
    return f"submission_{execution_id}.csv"


def submission_result_name(execution_id: str) -> str:
    return f"submission_result_{execution_id}.json"


def submission_csv_path(workspace_root: Path, execution_id: str) -> Path:
    return Path(workspace_root) / "artifacts" / submission_csv_name(execution_id)


def submission_result_path(workspace_root: Path, execution_id: str) -> Path:
    return Path(workspace_root) / "artifacts" / submission_result_name(execution_id)


def list_execution_submission_csvs(workspace_root: Path) -> list[Path]:
    artifacts = Path(workspace_root) / "artifacts"
    if not artifacts.is_dir():
        return []
    return sorted(artifacts.glob("submission_E-*.csv"))


class LeaderboardOutcome(BaseModel):
    public_score: float | None = None
    prior_public_score: float | None = None
    delta_vs_prior: float | None = None
    delta_vs_local: float | None = None
    scored_at: str | None = None
    submissions_url: str | None = None
    overfitting: bool | None = None


class ExecutionOutcomeSummary(BaseModel):
    """Rich local (+ optional LB) learning blob for one execution."""

    competition: str
    execution_id: str
    plan_id: str
    hypothesis_id: str | None = None
    execution_time_s: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    train_vs_validation: dict[str, Any] = Field(default_factory=dict)
    learning_gain: float | None = None
    learning_loss: float | None = None
    comparison: dict[str, Any] = Field(default_factory=dict)
    reflection: dict[str, Any] = Field(default_factory=dict)
    hypothesis_outcome: dict[str, Any] = Field(default_factory=dict)
    follow_up_hypothesis_id: str | None = None
    leaderboard: LeaderboardOutcome | None = None
    submission_path: str | None = None
    paths: dict[str, str] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def update_competition_skill_overlays(
    *,
    workspace_root: Path,
    summary: ExecutionOutcomeSummary,
    reflection: dict[str, Any] | None = None,
    techniques: list[str] | None = None,
) -> None:
    """Upsert per-run competition skill overlays (bounded / summarized on disk)."""
    try:
        from labpilot.research_engine.shared.skills import upsert_skill_overlay
    except Exception as exc:
        logger.warning("Skill overlay update skipped: %s", exc)
        return

    assessment = (reflection or {}).get("assessment") or {}
    lessons = (reflection or {}).get("lessons") or {}
    keep: list[str] = []
    avoid: list[str] = []
    try_next: list[str] = []
    if summary.learning_gain and summary.learning_gain > 0:
        keep.extend(techniques or [])
        if summary.hypothesis_id:
            keep.append(f"parent stack from {summary.hypothesis_id}")
    if summary.learning_loss and summary.learning_loss > 0:
        avoid.extend(techniques or [])
        avoid.append(f"regression on {summary.execution_id}")
    rec = str(assessment.get("recommendation") or "").strip()
    if rec:
        try_next.append(rec[:200])
    summary_lesson = str(
        lessons.get("summary") or assessment.get("summary") or ""
    ).strip()
    note = summary_lesson or str(
        (summary.hypothesis_outcome or {}).get("actual_outcome") or ""
    )
    lesson_id = f"{summary.execution_id}"
    for agent_key in (
        "code_engineer",
        "hypothesis_generator",
        "research_planner",
        "planning_engine",
        "experiment_reviewer",
        "research_brief",
    ):
        try:
            upsert_skill_overlay(
                workspace_root,
                agent_key,
                lesson_id=lesson_id,
                keep=keep,
                avoid=avoid,
                try_next=try_next,
                note=note[:400],
            )
        except Exception as exc:
            logger.warning("Skill overlay %s failed: %s", agent_key, exc)


def package_execution_submission(
    workspace_root: Path,
    execution_id: str,
) -> Path:
    """Ensure ``artifacts/submission_<E-id>.csv`` exists; also refresh latest convenience copy."""
    root = Path(workspace_root)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    scoped = submission_csv_path(root, execution_id)

    source = root / "submission.csv"
    if not source.is_file():
        pred = root / "predictions.csv"
        if pred.is_file():
            shutil.copy(pred, source)
        elif not scoped.is_file():
            source.write_text("id,prediction\n0,0\n", encoding="utf-8")

    if source.is_file():
        shutil.copy(source, scoped)
        # Convenience latest copy.
        shutil.copy(source, artifacts / "submission.csv")
    elif scoped.is_file():
        shutil.copy(scoped, artifacts / "submission.csv")
    else:
        raise FileNotFoundError(
            f"No submission.csv / predictions.csv to package for {execution_id}"
        )
    return scoped


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _primary_metric(metrics: dict[str, Any]) -> float | None:
    for key in (
        "cv_score",
        "cv_accuracy",
        "cv_rmse",
        "val_score",
        "val_loss",
        "accuracy",
        "rmse",
        "score",
    ):
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key])
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and key not in {"status", "execution_id"}:
            return float(value)
    return None


def _train_vs_validation(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "train_score",
        "train_loss",
        "train_accuracy",
        "val_score",
        "val_loss",
        "val_accuracy",
        "cv_score",
        "cv_accuracy",
        "cv_rmse",
    ):
        if key in metrics:
            out[key] = metrics[key]
    if "train_score" in out and "val_score" in out:
        try:
            out["train_val_gap"] = float(out["train_score"]) - float(out["val_score"])
        except (TypeError, ValueError):
            pass
    if "train_loss" in out and "val_loss" in out:
        try:
            out["train_val_loss_gap"] = float(out["val_loss"]) - float(out["train_loss"])
        except (TypeError, ValueError):
            pass
    return out


def _learning_deltas(comparison: dict[str, Any], metrics: dict[str, Any]) -> tuple[float | None, float | None]:
    delta = None
    for key in ("cv_delta", "primary_delta", "delta"):
        if key in comparison and isinstance(comparison[key], (int, float)):
            delta = float(comparison[key])
            break
    if delta is None:
        md = comparison.get("metric_deltas")
        if isinstance(md, dict):
            for val in md.values():
                if isinstance(val, (int, float)):
                    delta = float(val)
                    break
    if delta is None and isinstance(metrics.get("cv_delta"), (int, float)):
        delta = float(metrics["cv_delta"])
    if delta is None:
        return None, None
    if delta >= 0:
        return delta, None
    return None, abs(delta)


def _execution_time_s(execution: Any, workspace_root: Path, paths: ResearchPaths) -> float | None:
    started = getattr(execution, "started_at", None)
    completed = getattr(execution, "completed_at", None)
    if started and completed:
        try:
            return max(0.0, (completed - started).total_seconds())
        except Exception:
            pass
    # Fallback: sum task evidence durations.
    evidence_dir = paths.executions_dir / execution.id / "evidence"
    total = 0.0
    found = False
    if evidence_dir.is_dir():
        for path in evidence_dir.glob("*.json"):
            data = _load_json(path)
            meta = data.get("metadata") or {}
            if isinstance(meta.get("duration_s"), (int, float)):
                total += float(meta["duration_s"])
                found = True
    return total if found else None


def build_execution_outcome(
    *,
    knowledge_dir: Path,
    competition: str,
    execution: Any,
    plan: ResearchPlan,
    workspace_root: Path,
    reflection: dict[str, Any] | None = None,
) -> ExecutionOutcomeSummary:
    paths = ResearchPaths(knowledge_dir, competition)
    root = Path(workspace_root)
    metrics = _load_json(root / "metrics.json")
    comparison = _load_json(root / "comparison.json")
    # Also try artifacts/comparison.json (COMPARE historically wrote only there).
    if not comparison:
        comparison = _load_json(root / "artifacts" / "comparison.json")
    missing: list[str] = []
    if not metrics:
        missing.append("metrics")
    if not comparison:
        missing.append("comparison")

    train_val = _train_vs_validation(metrics)
    if not train_val:
        missing.append("train_vs_validation")
    gain, loss = _learning_deltas(comparison, metrics)
    if gain is None and loss is None:
        missing.append("learning_delta")

    exec_time = _execution_time_s(execution, root, paths)
    if exec_time is None:
        missing.append("execution_time_s")

    submission = submission_csv_path(root, execution.id)
    if not submission.is_file():
        missing.append("submission")

    assessment = (reflection or {}).get("assessment") or {}
    hyp_eval = (reflection or {}).get("hypothesis") or {}

    local_score = _primary_metric(metrics)
    actual = None
    if local_score is not None:
        actual = f"Local primary metric={local_score:.6g}"
        if gain is not None:
            actual += f"; learning_gain={gain:.6g}"
        if loss is not None:
            actual += f"; learning_loss={loss:.6g}"
        if train_val.get("train_val_gap") is not None:
            actual += f"; train_val_gap={train_val['train_val_gap']:.6g}"
    elif assessment.get("summary"):
        actual = str(assessment["summary"])

    summary = ExecutionOutcomeSummary(
        competition=competition,
        execution_id=execution.id,
        plan_id=plan.id,
        hypothesis_id=plan.hypothesis_id or None,
        execution_time_s=exec_time,
        metrics=metrics,
        train_vs_validation=train_val,
        learning_gain=gain,
        learning_loss=loss,
        comparison=comparison,
        reflection={
            "summary": assessment.get("summary"),
            "verdict": assessment.get("verdict") or assessment.get("hypothesis_outcome"),
            "recommendation": assessment.get("recommendation"),
            "likely_cause": assessment.get("likely_cause"),
            "cv_delta": assessment.get("cv_delta"),
        },
        hypothesis_outcome={
            "status": hyp_eval.get("status"),
            "why": hyp_eval.get("why"),
            "actual_outcome": actual,
            "public_score": None,
        },
        leaderboard=None,
        submission_path=str(submission) if submission.is_file() else None,
        paths={
            "metrics": str(root / "metrics.json"),
            "comparison": str(root / "comparison.json"),
            "submission": str(submission),
            "outcome": str(root / "artifacts" / "execution_outcome.json"),
        },
        missing=missing,
    )
    return summary


def write_outcome_files(
    summary: ExecutionOutcomeSummary,
    *,
    workspace_root: Path,
    paths: ResearchPaths,
) -> list[Path]:
    payload = summary.model_dump(mode="json")
    text = json.dumps(payload, indent=2) + "\n"
    written: list[Path] = []
    ws = Path(workspace_root) / "artifacts"
    ws.mkdir(parents=True, exist_ok=True)
    local = ws / "execution_outcome.json"
    local.write_text(text, encoding="utf-8")
    written.append(local)
    exec_dir = paths.executions_dir / summary.execution_id / "artifacts"
    exec_dir.mkdir(parents=True, exist_ok=True)
    exec_path = exec_dir / "execution_outcome.json"
    exec_path.write_text(text, encoding="utf-8")
    written.append(exec_path)
    return written


def upsert_experiment_artifact(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    techniques: list[str] | None = None,
) -> ResearchArtifact:
    tech = list(techniques or [])
    title = f"{competition} execution {summary.execution_id}"
    bits = []
    if summary.learning_gain is not None:
        bits.append(f"gain={summary.learning_gain:.4g}")
    if summary.learning_loss is not None:
        bits.append(f"loss={summary.learning_loss:.4g}")
    if summary.execution_time_s is not None:
        bits.append(f"time={summary.execution_time_s:.1f}s")
    if summary.leaderboard and summary.leaderboard.public_score is not None:
        bits.append(f"LB={summary.leaderboard.public_score:.6g}")
    artifact = ResearchArtifact(
        id=experiment_artifact_id(summary.execution_id),
        type=ResearchArtifactType.EXPERIMENT,
        source="labpilot",
        title=title,
        summary="; ".join(bits) or "Local experiment outcome",
        techniques=tech,
        claims=[
            c
            for c in [
                summary.hypothesis_outcome.get("actual_outcome"),
                (summary.reflection or {}).get("summary"),
            ]
            if c
        ],
        references=[summary.execution_id, summary.plan_id]
        + ([summary.hypothesis_id] if summary.hypothesis_id else []),
        confidence=0.6,
        competition_slug=competition,
        metadata=summary.model_dump(mode="json"),
    )
    with KnowledgeStore(knowledge_dir, competition) as store:
        store.upsert_artifact(artifact)
        for name in tech:
            tid = store.merge_technique(
                name,
                category="experiment",
                summary=f"Technique used in {summary.execution_id}",
                evidence=[artifact.id],
            )
            relation = "mentions"
            if summary.leaderboard and summary.leaderboard.overfitting:
                relation = "overfits"
            elif summary.learning_gain is not None and summary.learning_gain > 0:
                relation = "supports"
            elif summary.learning_loss is not None and summary.learning_loss > 0:
                relation = "contradicts"
            store.link_artifact_technique(artifact.id, tid, relation=relation)
    return artifact


def _techniques_from_plan(plan: ResearchPlan) -> list[str]:
    tags = list(plan.metadata.get("tags") or [])
    kind = plan.metadata.get("plan_kind")
    if kind:
        tags.append(str(kind))
    if plan.hypothesis_id:
        tags.append(f"hyp:{plan.hypothesis_id}")
    for key in ("technique",):
        val = plan.metadata.get(key)
        if val:
            tags.append(str(val))
    for key in ("combo_techniques", "technique_stack"):
        for item in plan.metadata.get(key) or []:
            tags.append(str(item))
    # Deduplicate preserving order; drop meta labels.
    _skip = {
        "baseline",
        "stacked",
        "combination",
        "ablation",
        "improvement",
        "technique",
    }
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        key = str(t).strip()
        if not key or key.lower() in _skip or key.lower().startswith("fork:"):
            continue
        if key in seen:
            continue
        # Expand joined combo labels so members are indexed separately.
        if "+" in key and key.count("+") <= 2 and " " not in key:
            for part in key.split("+"):
                part = part.strip()
                if part and part not in seen and part.lower() not in _skip:
                    seen.add(part)
                    out.append(part)
            continue
        seen.add(key)
        out.append(key)
    return out


_GENERIC_RECOMMENDATION_RE = re.compile(
    r"(?i)^(try the next|continue from|follow.?up after|post-submit|"
    r"iterate on the technique|baseline template|establish a registry)"
)


def _is_generic_recommendation(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 24:
        return True
    if _GENERIC_RECOMMENDATION_RE.search(cleaned):
        return True
    lowered = cleaned.lower()
    if "baseline" in lowered and "regular" not in lowered and "valid" not in lowered:
        # Baseline-like restatements are not improvement forks.
        if "improve" not in lowered and "gain" not in lowered:
            return True
    return False


def _already_covered_by_proposed(
    store: HypothesisStore,
    *,
    prediction: str,
    tags: list[str],
) -> bool:
    """Skip mint when an open proposed hyp already covers the same idea."""
    pred_tokens = {t for t in re.findall(r"[a-z0-9]+", prediction.lower()) if len(t) > 3}
    tag_set = {t.lower() for t in tags if t.lower() not in {"improvement", "follow-up", "execution", "submit"}}
    for hyp in store.list(status=HypothesisStatus.PROPOSED):
        existing_tags = {t.lower() for t in hyp.tags}
        if tag_set and tag_set & existing_tags:
            return True
        existing_tokens = {
            t for t in re.findall(r"[a-z0-9]+", (hyp.prediction + " " + hyp.observation).lower()) if len(t) > 3
        }
        if pred_tokens and len(pred_tokens & existing_tokens) >= max(3, len(pred_tokens) // 2):
            return True
    return False


def notify_proposed_hypotheses(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    note: str,
) -> list[str]:
    """Make open proposed hypotheses aware of the new experiment artifact."""
    store = HypothesisStore(knowledge_dir, competition)
    artifact_id = experiment_artifact_id(summary.execution_id)
    updated_ids: list[str] = []
    for hyp in store.list(status=HypothesisStatus.PROPOSED):
        if hyp.id == summary.hypothesis_id:
            continue
        try:
            store.annotate_experiment(
                hyp.id,
                execution_id=summary.execution_id,
                note=note,
                artifact_id=artifact_id,
            )
            updated_ids.append(hyp.id)
        except FileNotFoundError:
            continue
    return updated_ids


def maybe_mint_improvement_hypothesis(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    reflection: dict[str, Any] | None = None,
    overfitting: bool = False,
) -> str | None:
    """Mint a new hypothesis only when it is an improvement fork with expected gain.

    Skips generic / baseline-like follow-ups that add no benefit over existing
    proposed items. Successful runs with only a mild gain do **not** auto-mint;
    they update artifacts/beliefs and annotate proposed hypotheses instead.
    """
    store = HypothesisStore(knowledge_dir, competition)
    assessment = (reflection or {}).get("assessment") or {}
    hyp_eval = (reflection or {}).get("hypothesis") or {}

    parent = store.get(summary.hypothesis_id) if summary.hypothesis_id else None
    parent_conf = float(parent.confidence) if parent else 0.5

    observation = ""
    reason = ""
    prediction = ""
    expected_impact = 0.0
    confidence = parent_conf
    tags: list[str] = ["improvement"]

    if overfitting:
        observation = (
            f"Local CV looked strong on {summary.execution_id} but public LB did not"
        )
        reason = (
            "Overfit / validation gap: actual_outcome can look good while public_score "
            "regresses — fork toward generalization, not another baseline restatement."
        )
        prediction = (
            "Add regularization, simplify the model, or tighten validation so public "
            "score tracks local CV (expected LB recovery)."
        )
        expected_impact = max(
            0.01,
            abs(float(summary.leaderboard.delta_vs_prior))
            if summary.leaderboard and summary.leaderboard.delta_vs_prior is not None
            else 0.02,
        )
        confidence = min(0.9, parent_conf + 0.1)
        tags.extend(["overfitting", "generalization", summary.execution_id])
        # Only skip if backlog already has an overfit/generalization proposal.
        if _already_covered_by_proposed(
            store,
            prediction=prediction,
            tags=["overfitting", "generalization"],
        ):
            logger.info(
                "Skipping overfit mint for %s — generalization hyp already proposed",
                summary.execution_id,
            )
            return None
    else:
        loss = summary.learning_loss
        recommendation = str(
            assessment.get("recommendation")
            or hyp_eval.get("revised_prediction")
            or ""
        ).strip()

        # Mild success alone is not enough to mint — avoid baseline-like H-0xx noise.
        if loss is None or loss <= 0:
            return None
        if not recommendation or _is_generic_recommendation(recommendation):
            # Synthesize an actionable recovery prediction from the loss.
            recommendation = (
                f"Recover the {loss:.4g} local metric regression from "
                f"{summary.execution_id} with a distinct change (features, "
                f"regularization, or validation), not a baseline re-run."
            )

        expected_impact = float(loss)
        confidence = min(0.9, parent_conf + 0.05)
        reason = (
            f"Execution {summary.execution_id} lost {loss:.4g} vs prior; "
            f"fork an improvement with positive expected_impact, not a baseline clone."
        )
        observation = str(assessment.get("summary") or recommendation)[:500]
        prediction = recommendation[:500]
        tags.extend([summary.execution_id, "recover_loss"])
        if _already_covered_by_proposed(store, prediction=prediction, tags=tags):
            logger.info(
                "Skipping recovery mint for %s — already covered by proposed backlog",
                summary.execution_id,
            )
            return None

    if _is_generic_recommendation(prediction):
        return None
    if expected_impact <= 0:
        return None

    parent_id = parent.id if parent else summary.hypothesis_id or None
    if parent_id:
        tags.append(f"fork:{parent_id}")
    stack = list(parent.technique_stack) if parent else []
    if parent and parent.technique and parent.technique not in stack:
        stack.append(parent.technique)

    follow = store.create(
        observation=observation[:500] or prediction[:500],
        reason=reason[:1000],
        prediction=prediction[:500],
        confidence=confidence,
        expected_impact=expected_impact,
        tags=tags,
        source="reflection",
        created_by=HypothesisCreatedBy.REFLECTION,
        generator=HypothesisGenerator.RULE_ENGINE,
        origin=HypothesisOrigin.EXPERIMENT,
        evidence=[
            {
                "kind": "experiment",
                "ref": summary.execution_id,
                "note": "improvement fork from experiment outcome",
            }
        ],
        technique=parent.technique if parent else None,
        parent_hypothesis_id=parent_id,
        technique_stack=stack,
    )
    return follow.id


def maybe_mint_stacked_from_success(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    limit: int = 3,
) -> list[str]:
    """After a local gain, mint high-confidence stacked hyps from unused techniques."""
    if summary.learning_gain is None or summary.learning_gain <= 0:
        return []
    try:
        from labpilot.research_engine.intelligence.hypothesis.ledger import (
            build_experiment_ledger,
        )
    except Exception as exc:
        logger.warning("Stacked mint skipped (ledger): %s", exc)
        return []

    ledger = build_experiment_ledger(knowledge_dir, competition)
    parent_id = summary.hypothesis_id or ledger.winning_hypothesis_id
    if not parent_id:
        return []
    store = HypothesisStore(knowledge_dir, competition)
    parent = store.get(parent_id)
    stack = list(parent.technique_stack) if parent else list(ledger.winning_stack)
    if parent and parent.technique and parent.technique not in stack:
        stack.append(parent.technique)

    _skip_labels = {
        "baseline",
        "stacked",
        "combination",
        "ablation",
        "improvement",
        "technique",
        "untried",
        "unused_belief",
        "unused_claim",
        "belief",
        "pipeline_diff",
        "transfer",
        "failure_fix",
        "follow-up",
        "execution",
        "submit",
    }
    minted: list[str] = []
    for name in ledger.techniques_untried:
        if not name or name.strip().lower() in _skip_labels:
            continue
        if str(name).lower().startswith("fork:"):
            continue
        if ledger.is_failed(name):
            continue
        if any(
            h.technique == name and h.parent_hypothesis_id == parent_id
            for h in store.list()
            if h.status == HypothesisStatus.PROPOSED
        ):
            continue
        conf = min(
            0.95,
            float(parent.confidence if parent else 0.5)
            + 0.08
            + 0.05,
        )
        impact = max(0.005, float(summary.learning_gain) * 0.4)
        new_stack = [*stack, name] if name not in stack else list(stack)
        hyp = store.create(
            observation=(
                f"Parent {parent_id} gained {summary.learning_gain:.4g} on "
                f"{summary.execution_id}; unused technique {name} remains. "
                f"(technique {name})"
            ),
            reason=(
                f"Stack improvement: keep what worked on {parent_id} and merge {name} "
                f"(artifact experiment:{summary.execution_id}; technique {name})."
            ),
            prediction=(
                f"Adding {name} on top of {parent_id} will further improve the primary metric."
            ),
            confidence=conf,
            expected_impact=impact,
            tags=[name, "stacked", "improvement", f"fork:{parent_id}"],
            source="reflection",
            created_by=HypothesisCreatedBy.REFLECTION,
            generator=HypothesisGenerator.RULE_ENGINE,
            origin=HypothesisOrigin.EXPERIMENT,
            evidence=[
                {
                    "kind": "experiment",
                    "ref": summary.execution_id,
                    "note": "successful parent execution",
                }
            ],
            technique=name,
            parent_hypothesis_id=parent_id,
            technique_stack=new_stack,
        )
        minted.append(hyp.id)
        if len(minted) >= limit:
            break
    return minted


def _combo_members_from_hypothesis(hyp: Any) -> list[str]:
    if hyp is None:
        return []
    members = [str(t).strip() for t in (hyp.combo_techniques or []) if str(t).strip()]
    if len(members) >= 2:
        return members
    tech = str(hyp.technique or "").strip()
    tags_l = {str(t).lower() for t in (hyp.tags or [])}
    if "combination" in tags_l and "+" in tech:
        parts = [p.strip() for p in tech.split("+") if p.strip()]
        if len(parts) >= 2:
            return parts
    return members


def maybe_mint_ablation_from_combo_win(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
) -> list[str]:
    """On combination gain: mint leave-one-out ablation forks (no ablation on loss)."""
    if summary.learning_gain is None or summary.learning_gain <= 0:
        return []
    # Sparse ablation: only for material gains (≥ 0.01 absolute).
    if float(summary.learning_gain) < 0.01:
        return []
    if not summary.hypothesis_id:
        return []
    store = HypothesisStore(knowledge_dir, competition)
    parent = store.get(summary.hypothesis_id)
    members = _combo_members_from_hypothesis(parent)
    if len(members) < 2:
        return []

    stack = list(parent.technique_stack) if parent else []
    minted: list[str] = []
    for drop in members:
        kept = [m for m in members if m != drop]
        if not kept:
            continue
        label = "+".join(kept)
        if any(
            h.parent_hypothesis_id == parent.id
            and "ablation" in {t.lower() for t in h.tags}
            and set(h.combo_techniques or []) == set(kept)
            for h in store.list()
            if h.status == HypothesisStatus.PROPOSED
        ):
            continue
        hyp = store.create(
            observation=(
                f"Combination {parent.id} gained {summary.learning_gain:.4g}; "
                f"ablate by dropping `{drop}` to test if `{label}` alone suffices."
            ),
            reason=(
                f"Leave-one-out ablation after winning combo on {summary.execution_id}: "
                f"keep {kept}, drop {drop}."
            ),
            prediction=(
                f"Removing `{drop}` from the winning combo will show whether the gain "
                f"depends on that member or on {label}."
            ),
            confidence=min(0.9, float(parent.confidence) + 0.05),
            expected_impact=max(0.003, float(summary.learning_gain) * 0.25),
            tags=[*kept, "ablation", "stacked", "improvement", f"fork:{parent.id}"],
            source="reflection",
            created_by=HypothesisCreatedBy.REFLECTION,
            generator=HypothesisGenerator.RULE_ENGINE,
            origin=HypothesisOrigin.EXPERIMENT,
            evidence=[
                {
                    "kind": "experiment",
                    "ref": summary.execution_id,
                    "note": "ablation after combination win",
                }
            ],
            technique=label,
            parent_hypothesis_id=parent.id,
            technique_stack=stack,
            combo_techniques=kept,
        )
        minted.append(hyp.id)
    return minted


def maybe_mint_combo_from_success(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    llm_client: Any | None = None,
) -> list[str]:
    """After a gain, mint one LLM/rule-chosen combination hyp from the ledger shortlist."""
    if summary.learning_gain is None or summary.learning_gain <= 0:
        return []
    try:
        from labpilot.accessor.common.micro_agents import StructuredContext
        from labpilot.research_engine.intelligence.hypothesis.combo import (
            build_combo_shortlist,
            filter_picks_to_shortlist,
            picks_to_candidates,
            rule_engine_pick_combos,
        )
        from labpilot.research_engine.intelligence.hypothesis.ledger import (
            build_experiment_ledger,
        )
        from labpilot.research_engine.intelligence.micro_agents.artifacts import (
            ComboPortfolioDraft,
        )
        from labpilot.research_engine.intelligence.micro_agents.combo_portfolio import (
            ComboPortfolioAgent,
        )
    except Exception as exc:
        logger.warning("Combo mint skipped (import): %s", exc)
        return []

    ledger = build_experiment_ledger(knowledge_dir, competition)
    shortlist = build_combo_shortlist(ledger)
    if not shortlist:
        return []

    agent = ComboPortfolioAgent(llm_client=llm_client)
    draft = agent.run(
        StructuredContext(
            competition=competition,
            text="",
            data={
                "shortlist": shortlist,
                "limit": 1,
                "parent_stack": list(ledger.winning_stack),
                "parent_metrics": {},
                "avoid_pairs": [list(p) for p in ledger.avoid_pairs],
                "failed": list(ledger.techniques_failed),
                "skill_agent_key": "combo_portfolio",
            },
        )
    )
    picks_raw: list[dict[str, Any]] = []
    if isinstance(draft, ComboPortfolioDraft):
        picks_raw = [p.model_dump(mode="json") for p in draft.picks]
    picks = filter_picks_to_shortlist(picks_raw, shortlist) or rule_engine_pick_combos(
        shortlist, limit=1
    )
    if not picks:
        return []
    candidates = picks_to_candidates(picks[:1], ledger)
    if not candidates:
        return []

    store = HypothesisStore(knowledge_dir, competition)
    parent_id = summary.hypothesis_id or ledger.winning_hypothesis_id
    minted: list[str] = []
    for cand in candidates:
        techs = list(cand.metadata.get("combo_techniques") or [])
        if len(techs) < 2:
            continue
        if any(
            set(h.combo_techniques or []) == set(techs)
            and h.status == HypothesisStatus.PROPOSED
            for h in store.list()
        ):
            continue
        hyp = store.create(
            observation=cand.observation,
            reason=cand.reason,
            prediction=cand.prediction,
            confidence=cand.confidence,
            expected_impact=float(
                cand.metadata.get("expected_impact_value")
                or max(0.01, float(summary.learning_gain) * 0.5)
            ),
            tags=list(cand.tags),
            source="reflection",
            created_by=HypothesisCreatedBy.REFLECTION,
            generator=(
                HypothesisGenerator.LLM
                if agent.last_used_llm
                else HypothesisGenerator.RULE_ENGINE
            ),
            origin=HypothesisOrigin.EXPERIMENT,
            evidence=[
                {
                    "kind": "experiment",
                    "ref": summary.execution_id,
                    "note": "post-gain combination portfolio",
                }
            ],
            technique=cand.technique,
            parent_hypothesis_id=parent_id or cand.parent_hypothesis_id,
            technique_stack=list(cand.technique_stack),
            combo_techniques=techs,
        )
        minted.append(hyp.id)
    return minted


def record_combo_avoid_on_loss(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
) -> None:
    """On combination loss: ensure hyp is rejected so ledger records avoid_pairs.

    Ablation is never minted on loss.
    """
    if summary.learning_loss is None or summary.learning_loss <= 0:
        return
    if not summary.hypothesis_id:
        return
    store = HypothesisStore(knowledge_dir, competition)
    hyp = store.get(summary.hypothesis_id)
    members = _combo_members_from_hypothesis(hyp)
    if len(members) < 2:
        return
    # Status update usually happens via reflection; force REJECTED for combo losses
    # so avoid_pairs are groundable from the ledger.
    if hyp and hyp.status == HypothesisStatus.PROPOSED:
        try:
            store.update_outcome(
                hyp.id,
                actual_outcome=str(
                    (summary.hypothesis_outcome or {}).get("actual_outcome") or "loss"
                ),
                status=HypothesisStatus.REJECTED,
                evidence_run_id=summary.execution_id,
                why="Combination experiment lost; members recorded as avoid_pairs.",
            )
        except FileNotFoundError:
            pass
    logger.info(
        "Combo loss on %s — avoid_pairs for %s (no ablation)",
        summary.execution_id,
        members,
    )


def revalidate_outcome_claims(
    *,
    knowledge_dir: Path,
    competition: str,
) -> list[dict[str, Any]]:
    """Contest claims no measurement supports. Safe to call at any time.

    Separate from :func:`promote_outcome_claims` because repair must not depend
    on a *successful experiment*. Revalidation previously ran only inside
    `record_successful_execution`, so a campaign that completed no experiment —
    precisely when memory is most likely to be steering badly — never repaired
    itself. Measured 2026-08-07: a full campaign ran with 45 false `vit` claims
    intact because no execution succeeded to trigger the repair.
    """
    try:
        from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

        promoter = ClaimPromoter(knowledge_dir, competition)
        try:
            return promoter.revalidate_claims()
        finally:
            promoter.close()
    except Exception as exc:  # noqa: BLE001 — repair must never break a run
        logger.warning("Claim revalidation skipped: %s", exc)
        return []


def promote_outcome_claims(
    *,
    knowledge_dir: Path,
    competition: str,
    evidence_id: str | None = None,
) -> list[dict[str, Any]]:
    """Promote strong beliefs to claims after experiment/submit learning."""
    try:
        from labpilot.research_engine.reflection.claims.promoter import ClaimPromoter

        promoter = ClaimPromoter(knowledge_dir, competition)
        try:
            return promoter.promote_eligible(evidence_id=evidence_id)
        finally:
            promoter.close()
    except Exception as exc:
        logger.warning("Claim promotion skipped: %s", exc)
        return []


def update_hypothesis_from_local(
    *,
    knowledge_dir: Path,
    competition: str,
    summary: ExecutionOutcomeSummary,
    reflection: dict[str, Any] | None = None,
    llm_client: Any | None = None,
) -> str | None:
    """Update linked + proposed hyps from local outcome; mint only if worth trying.

    ``llm_client`` is threaded through to combo minting. Without it,
    ``ComboPortfolioAgent`` is built with no client and — since M14 phase 2a —
    raises rather than silently degrading, which aborts the whole
    learn-from-outcome step after a *successful* experiment.
    """
    store = HypothesisStore(knowledge_dir, competition)
    actual = (summary.hypothesis_outcome or {}).get("actual_outcome")
    hyp_eval = (reflection or {}).get("hypothesis") or {}
    status_raw = hyp_eval.get("status")
    status = None
    if status_raw:
        try:
            status = HypothesisStatus(str(status_raw))
        except ValueError:
            status = None

    if summary.hypothesis_id:
        try:
            store.update_outcome(
                summary.hypothesis_id,
                actual_outcome=str(actual) if actual else None,
                status=status,
                evidence_run_id=summary.execution_id,
                why=hyp_eval.get("why"),
            )
        except FileNotFoundError:
            logger.warning(
                "Linked hypothesis %s missing; skipping local update",
                summary.hypothesis_id,
            )

    note_bits = [f"New experiment {summary.execution_id}"]
    if summary.learning_gain is not None:
        note_bits.append(f"learning_gain={summary.learning_gain:.4g}")
    if summary.learning_loss is not None:
        note_bits.append(f"learning_loss={summary.learning_loss:.4g}")
    if actual:
        note_bits.append(str(actual)[:180])
    notify_proposed_hypotheses(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        note="; ".join(note_bits),
    )

    promote_outcome_claims(
        knowledge_dir=knowledge_dir,
        competition=competition,
        evidence_id=(reflection or {}).get("evidence", {}).get("id")
        if isinstance((reflection or {}).get("evidence"), dict)
        else None,
    )

    record_combo_avoid_on_loss(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
    )
    follow_id = maybe_mint_improvement_hypothesis(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        reflection=reflection,
        overfitting=bool(summary.leaderboard and summary.leaderboard.overfitting),
    )
    ablation_ids = maybe_mint_ablation_from_combo_win(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
    )
    combo_ids = maybe_mint_combo_from_success(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        llm_client=llm_client,
    )
    stacked_ids = maybe_mint_stacked_from_success(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        limit=2 if combo_ids else 3,
    )
    for group in (ablation_ids, combo_ids, stacked_ids):
        if group and not follow_id:
            follow_id = group[0]
            break
    return follow_id


def record_successful_execution(
    *,
    knowledge_dir: Path,
    competition: str,
    execution: Any,
    plan: ResearchPlan,
    workspace_root: Path,
    llm_client: Any | None = None,
) -> ExecutionOutcomeSummary:
    """Package submission, reflect if needed, write outcome + experiment artifact."""
    root = Path(workspace_root)
    package_execution_submission(root, execution.id)

    reflection: dict[str, Any] = {}
    try:
        from labpilot.research_engine.reflection.pipeline import run_reflection

        reflection = run_reflection(
            knowledge_dir,
            competition,
            execution_id=execution.id,
            workspace_path=root,
            plan_id=plan.id,
            hypothesis_id=plan.hypothesis_id or None,
            llm_client=llm_client,
            persist=True,
        )
    except Exception as exc:
        logger.warning("Reflection after success failed: %s", exc)
        reflection = {}

    # If inline upload already scored, fold LB into initial write.
    existing_lb = _load_json(submission_result_path(root, execution.id))
    if not existing_lb:
        existing_lb = _load_json(root / "artifacts" / "submission_result.json")

    summary = build_execution_outcome(
        knowledge_dir=knowledge_dir,
        competition=competition,
        execution=execution,
        plan=plan,
        workspace_root=root,
        reflection=reflection,
    )
    public_score = _extract_public_score(existing_lb)
    if public_score is not None:
        local = _primary_metric(summary.metrics)
        summary.leaderboard = LeaderboardOutcome(
            public_score=public_score,
            delta_vs_local=(public_score - local) if local is not None else None,
            scored_at=datetime.now(UTC).isoformat(),
            submissions_url=existing_lb.get("submissions_url"),
        )
        summary.hypothesis_outcome["public_score"] = public_score

    # Artifact + beliefs/techniques first so proposed hyps can reference them.
    paths = ResearchPaths(knowledge_dir, competition)
    techniques = _techniques_from_plan(plan)
    upsert_experiment_artifact(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        techniques=techniques,
    )

    follow_id = update_hypothesis_from_local(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        reflection=reflection,
        llm_client=llm_client,
    )
    summary.follow_up_hypothesis_id = follow_id
    summary.updated_at = datetime.now(UTC).isoformat()
    update_competition_skill_overlays(
        workspace_root=root,
        summary=summary,
        reflection=reflection,
        techniques=techniques,
    )

    write_outcome_files(summary, workspace_root=root, paths=paths)
    # Re-upsert so metadata includes follow_up id if minted.
    upsert_experiment_artifact(
        knowledge_dir=knowledge_dir,
        competition=competition,
        summary=summary,
        techniques=techniques,
    )
    return summary


def _extract_public_score(payload: dict[str, Any]) -> float | None:
    if isinstance(payload.get("public_score"), (int, float)):
        return float(payload["public_score"])
    upload = payload.get("upload") or {}
    if isinstance(upload.get("public_score"), (int, float)):
        return float(upload["public_score"])
    result = upload.get("result") or {}
    if isinstance(result, dict) and isinstance(result.get("public_score"), (int, float)):
        return float(result["public_score"])
    return None


def load_execution_outcome(
    workspace_root: Path,
    *,
    paths: ResearchPaths | None = None,
    execution_id: str | None = None,
) -> ExecutionOutcomeSummary | None:
    candidates = [
        Path(workspace_root) / "artifacts" / "execution_outcome.json",
    ]
    if paths is not None and execution_id:
        candidates.insert(
            0,
            paths.executions_dir / execution_id / "artifacts" / "execution_outcome.json",
        )
    for path in candidates:
        data = _load_json(path)
        if data.get("execution_id"):
            try:
                return ExecutionOutcomeSummary.model_validate(data)
            except Exception:
                continue
    return None
