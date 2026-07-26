"""EvidenceExtractor — deterministic metrics/config/runtime/comparison → SoR.

No LLM. Reads Engineer workspace + optional ``experiments`` comparison, then
persists an ``experiment_evidence`` row via :class:`ReflectionStore`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from labpilot.experiments.models import ExperimentComparison, Verdict
from labpilot.research_engine.execution.evidence import evidence_dir
from labpilot.research_engine.execution.store import ExecutionStore
from labpilot.research_engine.intelligence.paths import ResearchPaths
from labpilot.research_engine.reflection.store import ReflectionStore

logger = logging.getLogger(__name__)

EvidenceStrength = str  # strong | moderate | weak | rejected

# Absolute primary-metric improvement (signed for maximize) above noise → strong.
_DEFAULT_NOISE_EPSILON = 0.001
_DEFAULT_STRONG_DELTA = 0.01


def assess_strength(
    *,
    execution_failed: bool = False,
    metrics: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
    noise_epsilon: float = _DEFAULT_NOISE_EPSILON,
    strong_delta: float = _DEFAULT_STRONG_DELTA,
) -> EvidenceStrength:
    """Rule-based evidence strength for journal / critic input."""
    metrics = metrics or {}
    comparison = comparison or {}

    if execution_failed:
        return "rejected"

    if metrics.get("status") == "failed" or metrics.get("error"):
        return "rejected"

    verdict = comparison.get("verdict")
    if verdict == Verdict.REGRESSION.value or verdict == "regression":
        return "rejected"

    # Explicit Engineer comparison outcome.
    outcome = comparison.get("outcome")
    if outcome == "failed":
        return "rejected"

    signed = _signed_primary_delta(comparison, metrics)
    if signed is not None:
        if signed < -noise_epsilon:
            return "rejected"
        if signed >= strong_delta:
            return "strong"
        if abs(signed) <= noise_epsilon:
            return "weak"

    if verdict in {Verdict.WORTH_KEEPING.value, "worth_keeping"}:
        if signed is not None and signed > noise_epsilon:
            return "strong"
        return "moderate"

    if verdict in {Verdict.INCONCLUSIVE.value, "inconclusive"}:
        return "weak"

    if verdict in {Verdict.NOT_WORTH_KEEPING.value, "not_worth_keeping"}:
        return "weak"

    if outcome == "baseline":
        return "moderate" if metrics else "weak"

    if not metrics:
        return "weak"

    if signed is None and not verdict and not outcome:
        return "moderate"

    return "moderate"


def _signed_primary_delta(
    comparison: dict[str, Any], metrics: dict[str, Any]
) -> float | None:
    """Prefer comparison delta; else ``metrics`` nested delta fields."""
    if comparison.get("delta") is not None:
        try:
            return float(comparison["delta"])
        except (TypeError, ValueError):
            pass

    primary = comparison.get("primary_metric_key")
    deltas = comparison.get("metric_deltas") or {}
    if primary and primary in deltas:
        try:
            raw = float(deltas[primary])
        except (TypeError, ValueError):
            return None
        maximize = comparison.get("maximize", True)
        return raw if maximize else -raw

    # First numeric metric_deltas entry (stable sort).
    if isinstance(deltas, dict) and deltas:
        key = sorted(deltas.keys())[0]
        try:
            raw = float(deltas[key])
        except (TypeError, ValueError):
            return None
        maximize = comparison.get("maximize", True)
        return raw if maximize else -raw

    for key in ("primary_delta", "delta"):
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("skip unreadable json %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _comparison_to_dict(
    comparison: ExperimentComparison | dict[str, Any] | None,
) -> dict[str, Any]:
    if comparison is None:
        return {}
    if isinstance(comparison, ExperimentComparison):
        payload = comparison.model_dump(mode="json")
        # Signed direction unknown without maximize flag; treat raw delta as maximize.
        payload.setdefault("maximize", True)
        return payload
    return dict(comparison)


class EvidenceExtractor:
    """Build and optionally persist ``experiment_evidence`` from an execution."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.competition = competition
        self.paths = ResearchPaths(knowledge_dir, competition).ensure()
        self._store = ReflectionStore(knowledge_dir, competition)

    def close(self) -> None:
        self._store.close()

    @property
    def store(self) -> ReflectionStore:
        return self._store

    def extract(
        self,
        *,
        execution_id: str | None = None,
        workspace_path: Path | str | None = None,
        plan_id: str | None = None,
        hypothesis_id: str | None = None,
        experiment_id: str | None = None,
        comparison: ExperimentComparison | dict[str, Any] | None = None,
        execution_status: str | None = None,
        execution_error: str | None = None,
        persist: bool = True,
        noise_epsilon: float = _DEFAULT_NOISE_EPSILON,
        strong_delta: float = _DEFAULT_STRONG_DELTA,
    ) -> dict[str, Any]:
        """Extract structured evidence; persist when ``persist`` is True.

        Prefer ``workspace_path`` when known; otherwise resolves from
        ``research_executions`` when ``execution_id`` is set.
        """
        workspace, plan_id, hypothesis_id, experiment_id, exec_status, exec_error = (
            self._resolve_context(
                execution_id=execution_id,
                workspace_path=workspace_path,
                plan_id=plan_id,
                hypothesis_id=hypothesis_id,
                experiment_id=experiment_id,
                execution_status=execution_status,
                execution_error=execution_error,
            )
        )

        metrics = self._load_metrics(workspace)
        config_summary = self._load_config_summary(workspace)
        runtime_summary = self._load_runtime_summary(workspace, metrics)
        comparison_dict = self._load_comparison(workspace, comparison)
        task_pack = self._summarize_task_evidence(execution_id)

        failed = (
            exec_status == "failed"
            or bool(exec_error)
            or bool(task_pack.get("has_failure"))
        )
        strength = assess_strength(
            execution_failed=failed,
            metrics=metrics,
            comparison=comparison_dict,
            noise_epsilon=noise_epsilon,
            strong_delta=strong_delta,
        )

        metadata: dict[str, Any] = {
            "workspace_path": str(workspace) if workspace else None,
            "task_evidence": task_pack,
            "extractor": "rule_engine",
        }
        if exec_status:
            metadata["execution_status"] = exec_status
        if exec_error:
            metadata["execution_error"] = exec_error

        payload = {
            "execution_id": execution_id,
            "experiment_id": experiment_id,
            "plan_id": plan_id,
            "hypothesis_id": hypothesis_id,
            "metrics": metrics,
            "config_summary": config_summary,
            "runtime_summary": runtime_summary,
            "comparison": comparison_dict,
            "strength": strength,
            "metadata": metadata,
        }

        if not persist:
            return payload

        return self._store.create_evidence(**payload)

    def _resolve_context(
        self,
        *,
        execution_id: str | None,
        workspace_path: Path | str | None,
        plan_id: str | None,
        hypothesis_id: str | None,
        experiment_id: str | None,
        execution_status: str | None,
        execution_error: str | None,
    ) -> tuple[
        Path | None,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        workspace = Path(workspace_path) if workspace_path else None
        status = execution_status
        error = execution_error

        if execution_id and (
            workspace is None
            or plan_id is None
            or experiment_id is None
            or status is None
        ):
            exec_store = ExecutionStore(self.knowledge_dir, self.competition)
            try:
                execution = exec_store.get_execution(execution_id)
            finally:
                exec_store.close()
            if execution is not None:
                if workspace is None and execution.workspace_path:
                    workspace = Path(execution.workspace_path)
                plan_id = plan_id or execution.plan_id
                experiment_id = experiment_id or execution.experiment_id
                status = status or execution.status
                if error is None:
                    error = execution.error

        if plan_id and hypothesis_id is None:
            row = self._store._conn.execute(
                "SELECT hypothesis_id FROM research_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if row is not None and row["hypothesis_id"]:
                hypothesis_id = row["hypothesis_id"]

        return workspace, plan_id, hypothesis_id, experiment_id, status, error

    def _load_metrics(self, workspace: Path | None) -> dict[str, Any]:
        if workspace is None:
            return {}
        return _load_json(workspace / "metrics.json")

    def _load_config_summary(self, workspace: Path | None) -> dict[str, Any]:
        if workspace is None:
            return {}
        summary: dict[str, Any] = {}
        baseline = _load_json(workspace / "baseline_choice.json")
        if baseline:
            summary["baseline_choice"] = {
                k: baseline[k]
                for k in (
                    "problem_type",
                    "template_name",
                    "target_column",
                    "id_column",
                    "metric_name",
                )
                if k in baseline
            }
        overrides = _load_json(workspace / "training_overrides.json")
        if overrides:
            summary["training_overrides"] = overrides
        competition = _load_json(workspace / "competition.json")
        if competition.get("slug"):
            summary["competition_slug"] = competition["slug"]
        return summary

    def _load_runtime_summary(
        self, workspace: Path | None, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        runtime: dict[str, Any] = {}
        for key in ("runtime_seconds", "train_seconds", "elapsed_seconds"):
            if key in metrics and metrics[key] is not None:
                runtime[key] = metrics[key]
        if workspace is not None:
            record = _load_json(workspace / "runtime.json")
            if record:
                runtime["runtime_record"] = record
        return runtime

    def _load_comparison(
        self,
        workspace: Path | None,
        comparison: ExperimentComparison | dict[str, Any] | None,
    ) -> dict[str, Any]:
        if comparison is not None:
            return _comparison_to_dict(comparison)
        if workspace is None:
            return {}
        for rel in (
            Path("artifacts") / "comparison.json",
            Path("comparison.json"),
        ):
            data = _load_json(workspace / rel)
            if data:
                return data
        return {}

    def _summarize_task_evidence(self, execution_id: str | None) -> dict[str, Any]:
        if not execution_id:
            return {}
        directory = evidence_dir(self.paths, execution_id)
        if not directory.is_dir():
            return {}
        files = sorted(directory.glob("*.json"))
        failures: list[str] = []
        task_ids: list[str] = []
        for path in files:
            payload = _load_json(path)
            task_id = payload.get("task_id") or path.stem
            task_ids.append(str(task_id))
            if payload.get("passed") is False:
                failures.append(str(task_id))
        return {
            "task_count": len(files),
            "task_ids": task_ids,
            "failed_tasks": failures,
            "has_failure": bool(failures),
        }
