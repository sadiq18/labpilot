import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from labpilot.baseline.selector import BaselineChoice
from labpilot.competition.models import ProblemType
from labpilot.config import LLMConfig
from labpilot.improvement.models import (
    DEFAULT_IMPROVE_STAGES,
    ImprovementAction,
    ImprovementPlan,
    TrainingOverrides,
)
from labpilot.improvement.recipes import apply_recipes_from_profile
from labpilot.improvement.tuner import default_tabular_params, pick_tune_params
from labpilot.llm.client import LLMClient, create_llm_client
from labpilot.profiler.report import load_profile

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM = """You are a Kaggle experiment planner. Given a run's reflection and metrics,
output ONLY valid JSON matching this schema (no markdown fences):
{
  "strategy": "auto",
  "actions": ["tune_hyperparams"],
  "model_params": {"learning_rate": 0.05, "num_leaves": 31, "n_estimators": 300},
  "feature_recipes": [],
  "rationale": "short explanation"
}
Allowed actions: retrain, tune_hyperparams, apply_feature_recipe.
Allowed feature_recipes: target_encoding, log_numeric.
For tabular LightGBM, prefer tune_hyperparams when CV score has room to improve.
"""


class ImprovementPlanner:
    def __init__(self, config: LLMConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client if llm_client is not None else create_llm_client(config)

    def plan(
        self,
        parent_run_dir: Path,
        parent_run_id: str,
        strategy: str = "auto",
        *,
        random_seed: int = 42,
    ) -> tuple[ImprovementPlan, TrainingOverrides]:
        profile = load_profile(parent_run_dir)
        choice = BaselineChoice.model_validate_json(
            (parent_run_dir / "baseline_choice.json").read_text()
        )
        parent_overrides_path = parent_run_dir / "training_overrides.json"
        parent_params: dict[str, Any] = {}
        if parent_overrides_path.is_file():
            parent_params = json.loads(parent_overrides_path.read_text()).get("model_params", {})

        if strategy == "tune":
            return self._plan_tune(parent_run_id, choice, parent_params, random_seed=random_seed)
        if strategy == "features":
            return self._plan_features(parent_run_id, profile, choice, random_seed=random_seed)

        if self.llm_client is not None:
            try:
                return self._plan_auto_llm(
                    parent_run_dir,
                    parent_run_id,
                    profile,
                    choice,
                    parent_params,
                    random_seed=random_seed,
                )
            except Exception:
                logger.warning(
                    "LLM improvement planning failed; falling back to tune.",
                    exc_info=True,
                )

        return self._plan_tune(parent_run_id, choice, parent_params, random_seed=random_seed)

    def _plan_tune(
        self,
        parent_run_id: str,
        choice: BaselineChoice,
        parent_params: dict[str, Any],
        *,
        random_seed: int,
    ) -> tuple[ImprovementPlan, TrainingOverrides]:
        if not _is_tabular(choice.problem_type):
            params = default_tabular_params(random_seed=random_seed)
            rationale = "Non-tabular template: retrain with default params (tuning deferred)."
            actions = [ImprovementAction.RETRAIN]
        else:
            params = pick_tune_params(parent_params, random_seed=random_seed)
            rationale = "Deterministic LightGBM grid step from parent params."
            actions = [ImprovementAction.TUNE_HYPERPARAMS, ImprovementAction.RETRAIN]

        plan = ImprovementPlan(
            parent_run_id=parent_run_id,
            strategy="tune",
            actions=actions,
            model_params=params,
            feature_recipes=[],
            stages_to_run=list(DEFAULT_IMPROVE_STAGES),
            rationale=rationale,
        )
        overrides = TrainingOverrides(model_params=params)
        return plan, overrides

    def _plan_features(
        self,
        parent_run_id: str,
        profile,
        choice: BaselineChoice,
        *,
        random_seed: int,
    ) -> tuple[ImprovementPlan, TrainingOverrides]:
        recipes, te_cols, log_cols = apply_recipes_from_profile(
            profile, requested_recipes=["target_encoding", "log_numeric"]
        )
        params = default_tabular_params(random_seed=random_seed)
        actions = [ImprovementAction.APPLY_FEATURE_RECIPE, ImprovementAction.RETRAIN]
        if not recipes:
            actions = [ImprovementAction.RETRAIN]
        plan = ImprovementPlan(
            parent_run_id=parent_run_id,
            strategy="features",
            actions=actions,
            model_params=params,
            feature_recipes=recipes,
            stages_to_run=list(DEFAULT_IMPROVE_STAGES),
            rationale="Apply predefined tabular feature recipes from dataset profile.",
        )
        overrides = TrainingOverrides(
            model_params=params,
            feature_recipes=recipes,
            target_encoding_columns=te_cols,
            log_numeric_columns=log_cols,
        )
        return plan, overrides

    def _complete_with_retries(
        self, user: str, *, max_attempts: int, retry_delay_seconds: float
    ) -> str:
        """Same rationale as `llm.client.complete_with_fallback`'s retry support:
        free-tier providers frequently return transient errors that clear up
        within seconds, and a single blip shouldn't downgrade a real LLM plan
        to the deterministic `tune` fallback."""
        for attempt in range(1, max_attempts + 1):
            try:
                return self.llm_client.complete(_PLANNER_SYSTEM, user)
            except Exception:
                if attempt >= max_attempts:
                    raise
                logger.warning(
                    "LLM improvement planning call failed (attempt %d/%d); retrying in %.0fs.",
                    attempt,
                    max_attempts,
                    retry_delay_seconds,
                    exc_info=True,
                )
                time.sleep(retry_delay_seconds)
        raise AssertionError("unreachable")  # max_attempts >= 1 guaranteed by callers

    def _plan_auto_llm(
        self,
        parent_run_dir: Path,
        parent_run_id: str,
        profile,
        choice: BaselineChoice,
        parent_params: dict[str, Any],
        *,
        random_seed: int,
    ) -> tuple[ImprovementPlan, TrainingOverrides]:
        reflection_path = parent_run_dir / "reflection.md"
        metrics_path = parent_run_dir / "metrics.json"
        reflection = reflection_path.read_text() if reflection_path.is_file() else ""
        metrics = metrics_path.read_text() if metrics_path.is_file() else "{}"
        user = (
            f"Parent run: {parent_run_id}\n"
            f"Problem type: {choice.problem_type}\n"
            f"Template: {choice.template_name}\n"
            f"Parent model_params: {json.dumps(parent_params)}\n\n"
            f"Metrics:\n{metrics}\n\n"
            f"Reflection:\n{reflection[:8000]}"
        )
        raw = self._complete_with_retries(user, max_attempts=3, retry_delay_seconds=20.0)
        payload = _parse_json_object(raw)
        actions = [
            ImprovementAction(action)
            for action in payload.get("actions", [])
            if action in ImprovementAction._value2member_map_
        ]
        if not actions:
            actions = [ImprovementAction.TUNE_HYPERPARAMS, ImprovementAction.RETRAIN]

        model_params = payload.get("model_params") or {}
        if not model_params and ImprovementAction.TUNE_HYPERPARAMS in actions:
            model_params = pick_tune_params(parent_params, random_seed=random_seed)
        elif not model_params:
            model_params = default_tabular_params(random_seed=random_seed)
        else:
            model_params.setdefault("random_state", random_seed)
            model_params.setdefault("verbose", -1)

        feature_recipes = [
            recipe
            for recipe in payload.get("feature_recipes", [])
            if recipe in {"target_encoding", "log_numeric"}
        ]
        recipes, te_cols, log_cols = apply_recipes_from_profile(
            profile, requested_recipes=feature_recipes
        )

        plan = ImprovementPlan(
            parent_run_id=parent_run_id,
            strategy="auto",
            actions=actions,
            model_params=model_params,
            feature_recipes=recipes,
            stages_to_run=list(DEFAULT_IMPROVE_STAGES),
            rationale=str(payload.get("rationale", "LLM-generated improvement plan.")),
        )
        overrides = TrainingOverrides(
            model_params=model_params,
            feature_recipes=recipes,
            target_encoding_columns=te_cols,
            log_numeric_columns=log_cols,
        )
        return plan, overrides


def _is_tabular(problem_type: str) -> bool:
    return problem_type in {
        ProblemType.TABULAR_CLASSIFICATION.value,
        ProblemType.TABULAR_REGRESSION.value,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Planner response did not contain JSON object.")
    return json.loads(text[start : end + 1])
