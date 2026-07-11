import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.competition.models import CompetitionSpec
from labpilot.config import LLMConfig
from labpilot.profiler.tabular import DatasetProfile

logger = logging.getLogger(__name__)


class BriefGenerator:
    """Generate an AI research brief from competition + dataset profile."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            autoescape=select_autoescape(default=False),
        )

    def build_prompt(self, competition: CompetitionSpec, profile: DatasetProfile) -> str:
        system = (self.prompts_dir / "brief_system.md").read_text()
        user_template = self.env.get_template("brief_user.j2")
        user = user_template.render(competition=competition, profile=profile)
        return f"{system}\n\n---\n\n{user}"

    def generate(self, competition: CompetitionSpec, profile: DatasetProfile) -> str:
        prompt = self.build_prompt(competition, profile)
        # TODO: call LLM provider (OpenAI / Anthropic) using self.config
        logger.info(
            "Generating research brief for '%s' (LLM call not yet configured; using fallback).",
            competition.slug,
        )
        return self._fallback_brief(competition, profile, prompt)

    def save(self, run_dir: Path, competition: CompetitionSpec, profile: DatasetProfile) -> Path:
        brief = self.generate(competition, profile)
        output = run_dir / "brief.md"
        output.write_text(brief)
        logger.info("Saved research brief to %s", output)
        return output

    def _fallback_brief(
        self, competition: CompetitionSpec, profile: DatasetProfile, prompt: str
    ) -> str:
        return (
            f"# Research Brief: {competition.title or competition.slug}\n\n"
            f"> LLM generation not yet configured. Set OPENAI_API_KEY to enable.\n\n"
            f"## Problem Summary\n\n"
            f"Competition `{competition.slug}` with {profile.row_count} rows and "
            f"{profile.column_count} columns.\n\n"
            f"## Baseline Strategy\n\n"
            f"Start with a LightGBM tabular baseline using {self.config.model} for analysis.\n\n"
            f"## Prompt Preview\n\n"
            f"```\n{prompt[:500]}...\n```\n"
        )
