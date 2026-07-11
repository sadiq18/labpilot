import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.baseline.selector import BaselineChoice
from labpilot.config import LLMConfig
from labpilot.kaggle.client import SubmissionResult
from labpilot.profiler.tabular import DatasetProfile


class ReflectionGenerator:
    """Generate a post-run reflection with next-step recommendations."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
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
    ) -> str:
        template = self.env.get_template("reflection_user.j2")
        prompt = template.render(
            run_id=run_id,
            competition=competition,
            profile=profile,
            baseline=baseline,
            metrics=metrics,
            submission=submission,
        )
        # TODO: call LLM provider using self.config
        return self._fallback_reflection(run_id, competition, metrics, prompt)

    def save(self, run_dir: Path, content: str) -> Path:
        output = run_dir / "reflection.md"
        output.write_text(content)
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
        return (
            f"# Reflection: {competition}\n\n"
            f"**Run ID:** {run_id}\n\n"
            f"> LLM generation not yet configured. Set OPENAI_API_KEY to enable.\n\n"
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
