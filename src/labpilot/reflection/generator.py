import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.baseline.selector import BaselineChoice
from labpilot.config import LLMConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.llm.client import LLMClient, complete_with_fallback, resolve_llm_client
from labpilot.profiler.tabular import DatasetProfile
from labpilot.reflection.links import render_submission_links

logger = logging.getLogger(__name__)

_ENV_VAR_BY_PROVIDER = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class ReflectionGenerator:
    """Generate a post-run reflection with next-step recommendations."""

    def __init__(self, config: LLMConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client if llm_client is not None else resolve_llm_client(config)
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            autoescape=select_autoescape(default=False),
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
        system = (self.prompts_dir / "reflection_system.md").read_text()
        brief_text = ""
        if run_dir is not None:
            brief_path = run_dir / "brief.md"
            if brief_path.is_file():
                brief_text = brief_path.read_text()
        user = self.env.get_template("reflection_user.j2").render(
            run_id=run_id,
            competition=competition,
            profile=profile,
            baseline=baseline,
            metrics=metrics,
            submission=submission,
            brief_text=brief_text,
        )

        if self.llm_client is not None:
            logger.info("Generating reflection for run '%s' via LLM.", run_id)
            content = complete_with_fallback(self.config, system, user, self.llm_client)
            if content is not None:
                return content
            logger.warning(
                "LLM reflection generation failed for run '%s'; using fallback template text.",
                run_id,
            )
        else:
            logger.info(
                "Generating reflection for run '%s' (no LLM configured; using fallback).",
                run_id,
            )
        return self._fallback_reflection(run_id, competition, metrics, f"{system}\n\n---\n\n{user}")

    def save(self, run_dir: Path, content: str, submission: SubmissionResult | None = None) -> Path:
        if submission is not None:
            footer = render_submission_links(submission)
            if footer.strip() not in content:
                content = content.rstrip() + "\n\n" + footer
        output = run_dir / "reflection.md"
        output.write_text(content)
        logger.info("Saved reflection to %s", output)
        return output

    def load_metrics(self, run_dir: Path) -> dict[str, float]:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        data: dict[str, Any] = json.loads(metrics_path.read_text())
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    def _fallback_reflection(
        self, run_id: str, competition: str, metrics: dict[str, float], prompt: str
    ) -> str:
        metric_lines = "\n".join(f"- {k}: {v}" for k, v in metrics.items()) or "- none logged"
        env_var = _ENV_VAR_BY_PROVIDER.get(self.config.provider.strip().lower(), "OPENAI_API_KEY")
        return (
            f"# Reflection: {competition}\n\n"
            f"**Run ID:** {run_id}\n\n"
            f"> LLM generation not available. Set {env_var} to enable.\n\n"
            f"## Run Summary\n\n"
            f"Completed baseline pipeline for `{competition}`.\n\n"
            f"## Metrics\n\n{metric_lines}\n\n"
            f"## Recommended Next Steps\n\n"
            f"1. Review feature engineering opportunities from brief.md\n"
            f"2. Tune LightGBM hyperparameters (learning_rate, num_leaves)\n"
            f"3. Investigate CV vs leaderboard gap after submission\n"
            f"4. Add target encoding for high-cardinality categoricals\n"
            f"5. Ensemble with a linear model baseline\n"
        )
