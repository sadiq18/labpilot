"""Deterministic A/B experiment comparison (Milestone 2, Plan 3).

No LLM — categorizes config diffs, computes metric/runtime deltas, and emits
a threshold-based verdict plus a markdown view over the same structured fact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.accessor.common.derived import derived_note
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec
from labpilot.research_engine.shared.experiments.models import (
    ChangeCategory,
    ConfigChange,
    Experiment,
    ExperimentComparison,
    Verdict,
)

logger = logging.getLogger(__name__)

_CATEGORY_RULES: dict[str, ChangeCategory] = {
    "model_params.learning_rate": ChangeCategory.TRAINING_STRATEGY,
    "model_params.num_leaves": ChangeCategory.MODEL,
    "model_params.n_estimators": ChangeCategory.MODEL,
    "model_params.max_depth": ChangeCategory.MODEL,
    "model_params.min_child_samples": ChangeCategory.MODEL,
    "model_params.subsample": ChangeCategory.TRAINING_STRATEGY,
    "model_params.colsample_bytree": ChangeCategory.TRAINING_STRATEGY,
    "model_params.reg_alpha": ChangeCategory.TRAINING_STRATEGY,
    "model_params.reg_lambda": ChangeCategory.TRAINING_STRATEGY,
    "template_name": ChangeCategory.MODEL,
    "problem_type": ChangeCategory.OTHER,
}

# Recipe names that read as augmentations rather than feature engineering.
_AUGMENTATION_RECIPES: frozenset[str] = frozenset(
    {"mixup", "ema", "cutmix", "cutout", "randaugment", "autoaugment"}
)

_CATEGORY_HEADINGS: dict[ChangeCategory, str] = {
    ChangeCategory.MODEL: "Model",
    ChangeCategory.AUGMENTATION: "Augmentation",
    ChangeCategory.TRAINING_STRATEGY: "Training strategy",
    ChangeCategory.SCHEDULER: "Scheduler",
    ChangeCategory.FEATURE_ENGINEERING: "Feature engineering",
    ChangeCategory.OTHER: "Other",
}


def compare(
    base: Experiment,
    compare_exp: Experiment,
    *,
    noise_epsilon: float = 0.001,
    max_runtime_increase_pct: float = 50.0,
    primary_metric_key: str | None = None,
    maximize: bool = True,
    competition_dirs: tuple[Path, ...] = (),
) -> ExperimentComparison:
    """Compare two assembled experiments into a durable structured fact."""
    metric_deltas = _metric_deltas(base.metrics, compare_exp.metrics)
    changes = _config_changes(base, compare_exp)
    runtime_delta_seconds, runtime_delta_pct = _runtime_deltas(
        base.runtime_seconds, compare_exp.runtime_seconds
    )

    if primary_metric_key is not None:
        resolved_key = primary_metric_key
    else:
        resolved_key, maximize = resolve_primary_metric_key_and_direction(
            base, compare_exp, competition_dirs=competition_dirs
        )

    raw_delta: float | None = None
    if resolved_key is not None and resolved_key in metric_deltas:
        raw_delta = metric_deltas[resolved_key]
    elif resolved_key is not None:
        # Primary key present on only one side — treat as no shared metric.
        resolved_key = None

    signed_for_verdict = raw_delta
    if signed_for_verdict is not None and not maximize:
        signed_for_verdict = -signed_for_verdict

    verdict, reason = _verdict(
        signed_for_verdict,
        runtime_delta_pct,
        noise_epsilon=noise_epsilon,
        max_acceptable_runtime_increase_pct=max_runtime_increase_pct,
    )

    return ExperimentComparison(
        base_id=base.id,
        compare_id=compare_exp.id,
        primary_metric_key=resolved_key,
        metric_deltas=metric_deltas,
        changes=changes,
        runtime_delta_seconds=runtime_delta_seconds,
        runtime_delta_pct=runtime_delta_pct,
        verdict=verdict,
        verdict_reason=reason,
    )


def resolve_primary_metric_key_and_direction(
    base: Experiment,
    compare_exp: Experiment,
    *,
    competition_dirs: tuple[Path, ...] = (),
) -> tuple[str | None, bool]:
    """Pick primary metric key + maximize flag from competition.json when present.

    Prefer `cv_<MetricSpec.key>` when both sides have it, else bare `key`,
    else the first shared metric key (sorted). Direction defaults to maximize
    when no MetricSpec is found.
    """
    maximize = True
    spec_key: str | None = None
    for run_dir in competition_dirs:
        path = run_dir / "competition.json"
        if not path.is_file():
            continue
        try:
            spec = CompetitionSpec.model_validate_json(path.read_text())
        except (OSError, ValueError):
            continue
        if spec.evaluation_metric is not None:
            maximize = spec.evaluation_metric.direction != "minimize"
            spec_key = spec.evaluation_metric.key
            break

    shared = sorted(set(base.metrics) & set(compare_exp.metrics))
    if spec_key:
        cv_key = f"cv_{spec_key}"
        if cv_key in shared:
            return cv_key, maximize
        if spec_key in shared:
            return spec_key, maximize
    if shared:
        return shared[0], maximize
    return None, maximize


def render_markdown(comparison: ExperimentComparison) -> str:
    """Deterministic markdown view over an ExperimentComparison (no LLM).

    Unstamped: the stamp belongs to `write_comparison`, because this is also what
    `experiments compare --format markdown` prints from a live recomputation.
    """
    lines: list[str] = [
        f"# Comparison: {comparison.base_id} → {comparison.compare_id}",
        "",
        "## Changes",
        "",
    ]
    if not comparison.changes:
        lines.append("- (no config changes detected)")
    else:
        by_category: dict[ChangeCategory, list[ConfigChange]] = {}
        for change in comparison.changes:
            by_category.setdefault(change.category, []).append(change)
        for category in ChangeCategory:
            group = by_category.get(category)
            if not group:
                continue
            lines.append(f"### {_CATEGORY_HEADINGS[category]}")
            lines.append("")
            for change in group:
                lines.append(f"- {change.label}")
            lines.append("")
        if lines[-1] == "":
            lines.pop()

    lines.extend(["", "## Metrics", ""])
    if comparison.metric_deltas:
        for key in sorted(comparison.metric_deltas):
            delta = comparison.metric_deltas[key]
            marker = " (primary)" if key == comparison.primary_metric_key else ""
            lines.append(f"- `{key}`{marker}: {delta:+.4f}")
    else:
        lines.append("- (no shared numeric metrics)")

    if comparison.runtime_delta_seconds is None:
        lines.append("- Training time: not available")
    else:
        pct = comparison.runtime_delta_pct
        pct_part = f" ({pct:+.0f}%)" if pct is not None else ""
        lines.append(
            f"- Training time: {comparison.runtime_delta_seconds:+.1f}s{pct_part}"
        )
    lines.append("- Inference: not tracked")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"**{comparison.verdict.value}** — {comparison.verdict_reason}",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison(run_dir: Path, comparison: ExperimentComparison) -> None:
    """Persist comparison.json (source of truth) and comparison.md (view)."""
    (run_dir / "comparison.json").write_text(comparison.model_dump_json(indent=2) + "\n")
    # Stamped here and not in `render_markdown`: that renderer also serves
    # `experiments compare --format markdown`, which renders live, and a stamp
    # there would point that reader at a JSON which may not exist.
    (run_dir / "comparison.md").write_text(
        derived_note(
            source_of_record="comparison.json",
            warning=(
                "Written from the same object as the JSON beside it, so the two "
                "never disagree. `load_comparison` and every consumer read the "
                "JSON; edits here are lost on the next write."
            ),
        )
        + "\n\n"
        + render_markdown(comparison),
        encoding="utf-8",
    )


def load_comparison(run_dir: Path) -> ExperimentComparison | None:
    path = run_dir / "comparison.json"
    if not path.is_file():
        return None
    try:
        return ExperimentComparison.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        logger.debug("Could not load comparison from %s: %s", path, exc)
        return None


def _metric_deltas(
    base_metrics: dict[str, float], compare_metrics: dict[str, float]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(set(base_metrics) | set(compare_metrics)):
        base_value = base_metrics.get(key)
        compare_value = compare_metrics.get(key)
        if isinstance(base_value, (int, float)) and isinstance(compare_value, (int, float)):
            deltas[key] = float(compare_value) - float(base_value)
    return deltas


def _runtime_deltas(
    base_runtime: float | None, compare_runtime: float | None
) -> tuple[float | None, float | None]:
    if base_runtime is None or compare_runtime is None:
        return None, None
    delta = float(compare_runtime) - float(base_runtime)
    if base_runtime == 0:
        return delta, None
    pct = (delta / float(base_runtime)) * 100.0
    return delta, pct


def _config_changes(base: Experiment, compare_exp: Experiment) -> list[ConfigChange]:
    changes: list[ConfigChange] = []

    base_params = _flatten_dict(base.model_params, prefix="model_params")
    compare_params = _flatten_dict(compare_exp.model_params, prefix="model_params")
    for field in sorted(set(base_params) | set(compare_params)):
        base_value = base_params.get(field)
        compare_value = compare_params.get(field)
        if base_value == compare_value:
            continue
        short = field.removeprefix("model_params.")
        if field not in base_params:
            label = f"+ {short}={_format_value(compare_value)}"
        elif field not in compare_params:
            label = f"- {short} (was {_format_value(base_value)})"
        else:
            label = f"{short}: {_format_value(base_value)} → {_format_value(compare_value)}"
        changes.append(
            ConfigChange(
                category=_category_for_field(field),
                field=field,
                base_value=base_value,
                compare_value=compare_value,
                label=label,
            )
        )

    base_recipes = list(base.feature_recipes)
    compare_recipes = list(compare_exp.feature_recipes)
    for recipe in sorted(set(compare_recipes) - set(base_recipes)):
        changes.append(
            ConfigChange(
                category=_category_for_recipe(recipe),
                field="feature_recipes",
                base_value=None,
                compare_value=recipe,
                label=f"+ {recipe}",
            )
        )
    for recipe in sorted(set(base_recipes) - set(compare_recipes)):
        changes.append(
            ConfigChange(
                category=_category_for_recipe(recipe),
                field="feature_recipes",
                base_value=recipe,
                compare_value=None,
                label=f"- {recipe}",
            )
        )

    if base.template_name != compare_exp.template_name:
        changes.append(
            ConfigChange(
                category=_category_for_field("template_name"),
                field="template_name",
                base_value=base.template_name,
                compare_value=compare_exp.template_name,
                label=(
                    f"template_name: {_format_value(base.template_name)} → "
                    f"{_format_value(compare_exp.template_name)}"
                ),
            )
        )

    if base.problem_type != compare_exp.problem_type:
        changes.append(
            ConfigChange(
                category=_category_for_field("problem_type"),
                field="problem_type",
                base_value=base.problem_type,
                compare_value=compare_exp.problem_type,
                label=(
                    f"problem_type: {_format_value(base.problem_type)} → "
                    f"{_format_value(compare_exp.problem_type)}"
                ),
            )
        )

    return changes


def _flatten_dict(data: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        field = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=field))
        else:
            flat[field] = value
    return flat


def _category_for_field(field: str) -> ChangeCategory:
    if field in _CATEGORY_RULES:
        return _CATEGORY_RULES[field]
    # Prefix fallback: model_params.* defaults to MODEL unless a rule matched.
    if field.startswith("model_params."):
        return ChangeCategory.MODEL
    return ChangeCategory.OTHER


def _category_for_recipe(recipe: str) -> ChangeCategory:
    if recipe.lower() in _AUGMENTATION_RECIPES:
        return ChangeCategory.AUGMENTATION
    return ChangeCategory.FEATURE_ENGINEERING


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _verdict(
    metric_delta: float | None,
    runtime_delta_pct: float | None,
    *,
    noise_epsilon: float,
    max_acceptable_runtime_increase_pct: float,
) -> tuple[Verdict, str]:
    if metric_delta is None:
        return Verdict.INCONCLUSIVE, "No shared primary metric to compare."
    if abs(metric_delta) <= noise_epsilon:
        return (
            Verdict.INCONCLUSIVE,
            f"Metric delta ({metric_delta:+.4f}) within noise band (±{noise_epsilon}).",
        )
    if metric_delta < 0:
        return Verdict.REGRESSION, f"Primary metric regressed by {metric_delta:+.4f}."
    if (
        runtime_delta_pct is not None
        and runtime_delta_pct > max_acceptable_runtime_increase_pct
    ):
        return (
            Verdict.NOT_WORTH_KEEPING,
            f"Gain ({metric_delta:+.4f}) not worth +{runtime_delta_pct:.0f}% runtime.",
        )
    return Verdict.WORTH_KEEPING, f"Primary metric improved by {metric_delta:+.4f}."
