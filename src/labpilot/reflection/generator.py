"""Post-run reflection with structured JSON + markdown view (Milestone 2, Plan 4)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.baseline.selector import BaselineChoice
from labpilot.config import LLMConfig
from labpilot.experiments.models import (
    Experiment,
    ExperimentComparison,
    Hypothesis,
    StructuredReflection,
)
from labpilot.kaggle.client import SubmissionResult
from labpilot.llm.client import LLMClient, complete_with_fallback, resolve_llm_client
from labpilot.llm.json_utils import parse_json_object
from labpilot.profiler.tabular import DatasetProfile
from labpilot.reflection.links import render_submission_links

logger = logging.getLogger(__name__)

_ENV_VAR_BY_PROVIDER = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}

_DEFAULT_NEXT_STEPS = [
    "Review feature engineering opportunities from brief.md",
    "Tune LightGBM hyperparameters (learning_rate, num_leaves)",
    "Investigate CV vs leaderboard gap after submission",
    "Add target encoding for high-cardinality categoricals",
    "Ensemble with a linear model baseline",
]


class ReflectionGenerator:
    """Generate a post-run structured reflection with next-step recommendations."""

    def __init__(self, config: LLMConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client if llm_client is not None else resolve_llm_client(config)
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            autoescape=select_autoescape(default=False),
        )

    def generate_structured(
        self,
        *,
        experiment: Experiment,
        parent_experiment: Experiment | None,
        comparison: ExperimentComparison | None,
        hypothesis: Hypothesis | None,
        profile: DatasetProfile,
        baseline: BaselineChoice,
        metrics: dict[str, float],
        submission: SubmissionResult,
        run_dir: Path | None = None,
        max_new_hypotheses: int = 3,
        comparison_failed: bool = False,
    ) -> StructuredReflection:
        system = (self.prompts_dir / "reflection_system.md").read_text()
        brief_text = ""
        if run_dir is not None:
            brief_path = run_dir / "brief.md"
            if brief_path.is_file():
                brief_text = brief_path.read_text()

        user = self.env.get_template("reflection_user.j2").render(
            run_id=experiment.id,
            competition=experiment.competition,
            profile=profile,
            baseline=baseline,
            metrics=metrics,
            submission=submission,
            brief_text=brief_text,
            experiment=experiment,
            parent_experiment=parent_experiment,
            comparison=comparison,
            comparison_failed=comparison_failed,
            hypothesis=hypothesis,
            max_new_hypotheses=max_new_hypotheses,
        )

        if self.llm_client is not None:
            logger.info("Generating structured reflection for run '%s' via LLM.", experiment.id)
            content = complete_with_fallback(
                self.config, system, user, self.llm_client, max_attempts=3
            )
            if content is not None:
                try:
                    return self._parse_structured(
                        content,
                        run_id=experiment.id,
                        max_new_hypotheses=max_new_hypotheses,
                    )
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Failed to parse structured reflection for '%s': %s; using fallback.",
                        experiment.id,
                        exc,
                    )
            else:
                logger.warning(
                    "LLM reflection generation failed for run '%s'; using fallback.",
                    experiment.id,
                )
        else:
            logger.info(
                "Generating reflection for run '%s' (no LLM configured; using fallback).",
                experiment.id,
            )

        return self._fallback_structured(
            experiment=experiment,
            metrics=metrics,
            comparison=comparison,
            comparison_failed=comparison_failed,
            parent_experiment=parent_experiment,
        )

    def generate(
        self,
        run_id: str,
        competition: str,
        profile: DatasetProfile,
        baseline: BaselineChoice,
        metrics: dict[str, float],
        submission: SubmissionResult,
        run_dir: Path | None = None,
    ) -> str:
        """Legacy wrapper: structured reflection rendered to markdown."""
        from datetime import datetime

        experiment = Experiment(
            id=run_id,
            competition=competition,
            status="unknown",
            progress="",
            description="",
            metrics=metrics,
            created_at=datetime.now(),
        )
        structured = self.generate_structured(
            experiment=experiment,
            parent_experiment=None,
            comparison=None,
            hypothesis=None,
            profile=profile,
            baseline=baseline,
            metrics=metrics,
            submission=submission,
            run_dir=run_dir,
        )
        return render_markdown(structured)

    def save(
        self,
        run_dir: Path,
        content: str,
        submission: SubmissionResult | None = None,
    ) -> Path:
        if submission is not None:
            footer = render_submission_links(submission)
            if footer.strip() not in content:
                content = content.rstrip() + "\n\n" + footer
        output = run_dir / "reflection.md"
        output.write_text(content)
        logger.info("Saved reflection to %s", output)
        return output

    def save_structured(
        self,
        run_dir: Path,
        structured: StructuredReflection,
        submission: SubmissionResult | None = None,
    ) -> list[Path]:
        json_path = run_dir / "reflection.json"
        json_path.write_text(structured.model_dump_json(indent=2) + "\n")
        md = render_markdown(structured)
        md_path = self.save(run_dir, md, submission=submission)
        return [json_path, md_path]

    def load_metrics(self, run_dir: Path) -> dict[str, float]:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        data: dict[str, Any] = json.loads(metrics_path.read_text())
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    def _parse_structured(
        self,
        content: str,
        *,
        run_id: str,
        max_new_hypotheses: int,
    ) -> StructuredReflection:
        payload = parse_json_object(content)
        payload["run_id"] = run_id
        payload["generated_by"] = "llm"
        drafts = payload.get("new_hypotheses") or []
        if isinstance(drafts, list) and len(drafts) > max_new_hypotheses:
            payload["new_hypotheses"] = drafts[:max_new_hypotheses]
        return StructuredReflection.model_validate(payload)

    def _fallback_structured(
        self,
        *,
        experiment: Experiment,
        metrics: dict[str, float],
        comparison: ExperimentComparison | None,
        comparison_failed: bool,
        parent_experiment: Experiment | None,
    ) -> StructuredReflection:
        env_var = _ENV_VAR_BY_PROVIDER.get(self.config.provider.strip().lower(), "OPENAI_API_KEY")
        metric_lines = [f"{k}: {v}" for k, v in metrics.items()] or ["none logged"]

        if comparison_failed and parent_experiment is not None:
            observation = (
                f"Reflection for `{experiment.id}` could not use parent/child comparison "
                f"(parent `{parent_experiment.id}`). LLM unavailable ({env_var})."
            )
            likely_cause = (
                "Comparison context failed or was unavailable; treat this as a degraded "
                "reflection. Fix comparison inputs or resume the pipeline, then re-run "
                "write_reflection."
            )
            suggested = [
                "Inspect parent/child run artifacts and re-run `research resume` if stages failed",
                "Ensure metrics.json and training overrides exist on both runs",
                "Re-run write_reflection after comparison.json can be assembled",
            ]
        elif not metrics and parent_experiment is not None:
            observation = (
                f"Run `{experiment.id}` reached reflection with no numeric metrics logged."
            )
            likely_cause = (
                "Training/evaluation may have failed or been skipped; resume after fixing "
                "the failing stage."
            )
            suggested = [
                "Check train_model / evaluate_cv stage errors in the manifest",
                "Fix the root cause and `research resume --run-id "
                f"{experiment.id}`",
                "Re-check metrics.json before interpreting results",
            ]
        else:
            observation = (
                f"Completed pipeline for `{experiment.competition}` "
                f"(run `{experiment.id}`). LLM generation not available — set {env_var}."
            )
            likely_cause = "Template fallback only; no LLM analysis was performed."
            if comparison is not None:
                observation += (
                    f" Comparison vs `{comparison.base_id}`: "
                    f"{comparison.verdict.value} — {comparison.verdict_reason}"
                )
            suggested = list(_DEFAULT_NEXT_STEPS)

        return StructuredReflection(
            run_id=experiment.id,
            observation=observation,
            evidence=metric_lines,
            likely_cause=likely_cause,
            confidence=0.0,
            suggested_next=suggested,
            hypothesis_updates=[],
            new_hypotheses=[],
            generated_by="template_fallback",
        )


def render_markdown(structured: StructuredReflection) -> str:
    """Deterministic markdown view over StructuredReflection (no second LLM call)."""
    lines: list[str] = [
        f"# Reflection: {structured.run_id}",
        "",
        "## Observation",
        "",
        structured.observation,
        "",
        "## Evidence",
        "",
    ]
    if structured.evidence:
        for item in structured.evidence:
            lines.append(f"- {item}")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Likely cause",
            "",
            structured.likely_cause,
            "",
            f"**Confidence:** {structured.confidence:.2f}",
            "",
            "## Suggested next steps",
            "",
        ]
    )
    if structured.suggested_next:
        for index, step in enumerate(structured.suggested_next, start=1):
            lines.append(f"{index}. {step}")
    else:
        lines.append("- (none)")

    if structured.hypothesis_updates:
        lines.extend(["", "## Hypothesis updates", ""])
        for update in structured.hypothesis_updates:
            note = f" — {update.note}" if update.note else ""
            lines.append(
                f"- `{update.hypothesis_id}` → `{update.new_status.value}`{note}"
            )

    if structured.new_hypotheses:
        lines.extend(["", "## New hypotheses proposed", ""])
        for draft in structured.new_hypotheses:
            tags = f" [{', '.join(draft.tags)}]" if draft.tags else ""
            lines.append(
                f"- {draft.prediction} (confidence {draft.confidence:.2f}){tags}"
            )
            lines.append(f"  - Observation: {draft.observation}")
            lines.append(f"  - Reason: {draft.reason}")

    lines.extend(
        [
            "",
            f"_Generated by: `{structured.generated_by}`_",
            "",
        ]
    )
    return "\n".join(lines)


# Re-export for tests
__all__ = [
    "ReflectionGenerator",
    "render_markdown",
]
