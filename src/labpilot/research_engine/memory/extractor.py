"""ExperienceExtractor — deterministic experiment (+ reflection) → ExperienceRecord.

No LLM. Reads existing SoR (experiment / plan / hypothesis / reflection) and
persists via :class:`ExperienceStore`. Facets come from :class:`FacetPipeline`
(Stage 2 artifact-aware extractors; not treated as ground truth).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from labpilot.research_engine.memory.facets import FacetContext, FacetPipeline
from labpilot.research_engine.memory.models import (
    ExperienceArtifacts,
    ExperienceOutcome,
    ExperienceRecord,
)
from labpilot.research_engine.memory.store import ExperienceStore
from labpilot.research_engine.planner.store import PlanStore
from labpilot.research_engine.reflection.evidence.extractor import (
    _signed_primary_delta,
    assess_strength,
)
from labpilot.research_engine.shared.experiments.hypothesis import HypothesisStore
from labpilot.research_engine.shared.experiments.models import (
    Experiment,
    StructuredReflection,
    Verdict,
)
from labpilot.workspace import competition_workspace_path


class ExperienceExtractor:
    """Map a completed experiment (+ optional reflection) into an Experience Record."""

    def __init__(
        self,
        knowledge_dir: Path,
        *,
        store: ExperienceStore | None = None,
        facet_pipeline: FacetPipeline | None = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self._store = store or ExperienceStore(self.knowledge_dir)
        self._owns_store = store is None
        self._facet_pipeline = facet_pipeline or FacetPipeline()

    def close(self) -> None:
        if self._owns_store:
            self._store.close()

    @property
    def store(self) -> ExperienceStore:
        return self._store

    def extract(
        self,
        *,
        competition: str,
        experiment: Experiment | dict[str, Any] | None = None,
        experiment_id: str | None = None,
        execution_id: str | None = None,
        plan_id: str | None = None,
        hypothesis_id: str | None = None,
        reflection: StructuredReflection | dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        workspace_path: Path | str | None = None,
        persist: bool = True,
    ) -> ExperienceRecord:
        """Build an Experience Record; upsert when ``persist`` is True.

        Idempotency key prefers ``experiment_id``, then ``execution_id``.
        """
        payload = self._normalize_experiment(
            competition=competition,
            experiment=experiment,
            experiment_id=experiment_id,
            execution_id=execution_id,
            plan_id=plan_id,
            hypothesis_id=hypothesis_id,
            workspace_path=workspace_path,
        )
        reflection_payload = self._normalize_reflection(
            reflection, experiment=payload.get("_experiment_model")
        )
        comparison_dict = dict(comparison or payload.get("comparison") or {})

        exp_id = payload.get("experiment_id")
        exec_id = payload.get("execution_id")
        idempotency_key = str(exp_id or exec_id or "").strip()
        if not idempotency_key:
            raise ValueError("experiment_id or execution_id is required for experience extract")

        hyp_id = payload.get("hypothesis_id")
        hypothesis_text = self._hypothesis_text(competition, hyp_id, payload, reflection_payload)
        goal = self._goal_text(competition, payload.get("plan_id"), payload)
        action = self._action_text(payload, reflection_payload)
        result = self._result_text(payload, comparison_dict, reflection_payload)
        outcome = self._outcome(
            payload=payload,
            comparison=comparison_dict,
            reflection=reflection_payload,
        )
        workspace = (
            Path(workspace_path)
            if workspace_path
            else competition_workspace_path(self.knowledge_dir, competition)
        )
        facets = self._facet_pipeline.extract(
            FacetContext(
                competition=competition,
                payload=payload,
                hypothesis_text=hypothesis_text,
                action=action,
                reflection=reflection_payload,
                comparison=comparison_dict,
                workspace_path=workspace if workspace.is_dir() else None,
                paper_texts=_paper_texts(payload),
            )
        )
        artifacts = ExperienceArtifacts(
            experiment_id=str(exp_id) if exp_id else None,
            execution_id=str(exec_id) if exec_id else None,
            plan_id=payload.get("plan_id"),
            reflection_id=reflection_payload.get("id") or reflection_payload.get("run_id"),
            git_commit=payload.get("git_commit"),
            metrics=dict(payload.get("metrics") or {}),
            comparison=comparison_dict,
        )

        now = datetime.now(UTC)
        existing = self._store.get_by_idempotency_key(idempotency_key)
        record = ExperienceRecord(
            id=existing.id if existing else self._store.new_experience_id(),
            source_competition=competition,
            goal=goal,
            hypothesis=hypothesis_text,
            hypothesis_id=str(hyp_id) if hyp_id else None,
            action=action,
            result=result,
            outcome=outcome,
            artifacts=artifacts,
            facets=facets,
            idempotency_key=idempotency_key,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        if not persist:
            return record
        return self._store.upsert(record)

    def _normalize_experiment(
        self,
        *,
        competition: str,
        experiment: Experiment | dict[str, Any] | None,
        experiment_id: str | None,
        execution_id: str | None,
        plan_id: str | None,
        hypothesis_id: str | None,
        workspace_path: Path | str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "competition": competition,
            "experiment_id": experiment_id,
            "execution_id": execution_id,
            "plan_id": plan_id,
            "hypothesis_id": hypothesis_id,
            "metrics": {},
            "description": "",
            "status": "",
            "problem_type": None,
            "git_commit": None,
            "feature_recipes": [],
            "comparison": {},
        }

        if isinstance(experiment, Experiment):
            payload["_experiment_model"] = experiment
            payload["experiment_id"] = experiment.id
            payload["hypothesis_id"] = experiment.hypothesis_id or hypothesis_id
            payload["metrics"] = dict(experiment.metrics or {})
            payload["description"] = experiment.description or ""
            payload["status"] = experiment.status or ""
            payload["problem_type"] = experiment.problem_type
            payload["git_commit"] = experiment.git_commit
            payload["feature_recipes"] = list(experiment.feature_recipes or [])
            payload["model_params"] = dict(experiment.model_params or {})
        elif isinstance(experiment, dict):
            payload.update({k: v for k, v in experiment.items() if v is not None})
            if experiment.get("experiment_id") and not payload.get("experiment_id"):
                payload["experiment_id"] = experiment["experiment_id"]
            if experiment.get("id") and not payload.get("experiment_id"):
                payload["experiment_id"] = experiment["id"]
            if experiment.get("run_id") and not payload.get("experiment_id"):
                payload["experiment_id"] = experiment["run_id"]
            metrics = experiment.get("metrics")
            if isinstance(metrics, dict):
                payload["metrics"] = metrics

        # Fill gaps from workspace agent record when ids are known.
        workspace = (
            Path(workspace_path)
            if workspace_path
            else competition_workspace_path(self.knowledge_dir, competition)
        )
        lookup_id = payload.get("experiment_id") or payload.get("execution_id")
        if lookup_id and workspace is not None:
            from labpilot.research_engine.agents.git_evolution import (
                find_experiment_record,
            )

            disk = find_experiment_record(workspace, str(lookup_id))
            if isinstance(disk, dict):
                for key in (
                    "experiment_id",
                    "execution_id",
                    "plan_id",
                    "git_commit",
                    "status",
                    "metrics",
                    "files_changed",
                ):
                    if payload.get(key) in (None, "", {}, []) and disk.get(key) not in (
                        None,
                        "",
                        {},
                        [],
                    ):
                        payload[key] = disk[key]
                if not payload.get("experiment_id") and disk.get("experiment_id"):
                    payload["experiment_id"] = disk["experiment_id"]
                if not payload.get("execution_id") and disk.get("execution_id"):
                    payload["execution_id"] = disk["execution_id"]

        if plan_id and not payload.get("plan_id"):
            payload["plan_id"] = plan_id
        if hypothesis_id and not payload.get("hypothesis_id"):
            payload["hypothesis_id"] = hypothesis_id
        if execution_id and not payload.get("execution_id"):
            payload["execution_id"] = execution_id
        if experiment_id and not payload.get("experiment_id"):
            payload["experiment_id"] = experiment_id

        return payload

    def _normalize_reflection(
        self,
        reflection: StructuredReflection | dict[str, Any] | None,
        *,
        experiment: Experiment | None,
    ) -> dict[str, Any]:
        if reflection is None and experiment is not None and experiment.reflection is not None:
            reflection = experiment.reflection
        if reflection is None:
            return {}
        if isinstance(reflection, StructuredReflection):
            return reflection.model_dump(mode="json")
        return dict(reflection)

    def _hypothesis_text(
        self,
        competition: str,
        hyp_id: str | None,
        payload: dict[str, Any],
        reflection: dict[str, Any],
    ) -> str:
        if payload.get("hypothesis"):
            return str(payload["hypothesis"]).strip()
        if hyp_id:
            hyp = HypothesisStore(self.knowledge_dir, competition).get(str(hyp_id))
            if hyp is not None:
                parts = [hyp.prediction or "", hyp.reason or "", hyp.observation or ""]
                text = " | ".join(p for p in parts if p).strip()
                if text:
                    return text
                if hyp.technique:
                    return f"technique={hyp.technique}"
        obs = reflection.get("observation")
        if obs:
            return str(obs).strip()
        return ""

    def _goal_text(
        self,
        competition: str,
        plan_id: str | None,
        payload: dict[str, Any],
    ) -> str:
        if payload.get("goal"):
            return str(payload["goal"]).strip()
        if plan_id:
            store = PlanStore(self.knowledge_dir, competition)
            try:
                plan = store.get_plan(str(plan_id))
            finally:
                store.close()
            if plan is not None and plan.goal:
                return plan.goal.strip()
        description = str(payload.get("description") or "").strip()
        if description:
            return description
        return f"Improve {competition}"

    def _action_text(
        self,
        payload: dict[str, Any],
        reflection: dict[str, Any],
    ) -> str:
        if payload.get("action"):
            return str(payload["action"]).strip()
        chunks: list[str] = []
        description = str(payload.get("description") or "").strip()
        if description:
            chunks.append(description)
        files = payload.get("files_changed") or []
        if isinstance(files, list) and files:
            chunks.append("files: " + ", ".join(str(f) for f in files[:8]))
        recipes = payload.get("feature_recipes") or []
        if isinstance(recipes, list) and recipes:
            chunks.append("features: " + ", ".join(str(r) for r in recipes[:6]))
        cause = reflection.get("likely_cause")
        if cause and not chunks:
            chunks.append(str(cause))
        return "; ".join(chunks)

    def _result_text(
        self,
        payload: dict[str, Any],
        comparison: dict[str, Any],
        reflection: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        signed = _signed_primary_delta(comparison, payload.get("metrics") or {})
        if signed is not None:
            parts.append(f"delta={signed:+.6g}")
        metrics = payload.get("metrics") or {}
        if isinstance(metrics, dict) and metrics:
            # Prefer common score keys; else first few numeric metrics.
            preferred = ("cv_score", "lb_score", "public_score", "score", "primary")
            shown: list[str] = []
            for key in preferred:
                if key in metrics and _is_number(metrics[key]):
                    shown.append(f"{key}={metrics[key]}")
            if not shown:
                for key, value in list(metrics.items())[:4]:
                    if _is_number(value):
                        shown.append(f"{key}={value}")
            if shown:
                parts.append(", ".join(shown))
        if comparison.get("verdict"):
            parts.append(f"verdict={comparison['verdict']}")
        if reflection.get("observation") and not parts:
            parts.append(str(reflection["observation"])[:240])
        status = str(payload.get("status") or "").strip()
        if status and status not in {"completed", "success"}:
            parts.append(f"status={status}")
        return "; ".join(parts)

    def _outcome(
        self,
        *,
        payload: dict[str, Any],
        comparison: dict[str, Any],
        reflection: dict[str, Any],
    ) -> ExperienceOutcome:
        status = str(payload.get("status") or "").lower()
        if status in {"failed", "error", "cancelled"}:
            return "fail"
        if comparison.get("outcome") == "failed":
            return "fail"
        verdict = comparison.get("verdict")
        if verdict in {Verdict.REGRESSION.value, "regression"}:
            return "fail"
        if verdict in {Verdict.WORTH_KEEPING.value, "worth_keeping"}:
            return "success"
        if verdict in {Verdict.NOT_WORTH_KEEPING.value, "not_worth_keeping"}:
            return "fail"

        strength = assess_strength(
            execution_failed=status in {"failed", "error"},
            metrics=payload.get("metrics") or {},
            comparison=comparison,
        )
        if strength in {"strong", "moderate"}:
            signed = _signed_primary_delta(comparison, payload.get("metrics") or {})
            if signed is not None:
                return "success" if signed > 0 else "fail"
            if strength == "strong":
                return "success"

        # Reflection hypothesis updates: any confirmed → success hint.
        for update in reflection.get("hypothesis_updates") or []:
            new_status = str(
                update.get("new_status") if isinstance(update, dict) else getattr(update, "new_status", "")
            ).lower()
            if new_status in {"confirmed", "success"}:
                return "success"
            if new_status in {"rejected"}:
                return "fail"

        if status in {"completed", "success", "succeeded"}:
            signed = _signed_primary_delta(comparison, payload.get("metrics") or {})
            if signed is not None:
                return "success" if signed >= 0 else "fail"
            return "success"
        return "fail"


def _paper_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    papers = payload.get("papers")
    if isinstance(papers, list):
        for item in papers:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(str(item.get("title") or ""))
                texts.append(str(item.get("abstract") or item.get("summary") or ""))
    return [t for t in texts if t.strip()]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
