import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.brief.context import render_competition_context
from labpilot.competition.models import CompetitionSpec
from labpilot.config import LLMConfig
from labpilot.llm.client import LLMClient, complete_with_fallback, resolve_llm_client
from labpilot.profiler.tabular import DatasetProfile

logger = logging.getLogger(__name__)

_ENV_VAR_BY_PROVIDER = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}


class BriefGenerator:
    """Generate an AI research brief from competition + dataset profile."""

    def __init__(self, config: LLMConfig, llm_client: LLMClient | None = None) -> None:
        self.config = config
        self.llm_client = llm_client if llm_client is not None else resolve_llm_client(config)
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
        system = (self.prompts_dir / "brief_system.md").read_text()
        user = self.env.get_template("brief_user.j2").render(
            competition=competition, profile=profile
        )

        if self.llm_client is not None:
            logger.info("Generating research brief for '%s' via LLM.", competition.slug)
            narrative = complete_with_fallback(
                self.config, system, user, self.llm_client, max_attempts=3
            )
            if narrative is not None:
                return render_competition_context(competition) + narrative
            logger.warning(
                "LLM brief generation failed for '%s'; using fallback template text.",
                competition.slug,
            )
        else:
            logger.info(
                "Generating research brief for '%s' (no LLM configured; using fallback).",
                competition.slug,
            )
        return self._fallback_brief(competition, profile, f"{system}\n\n---\n\n{user}")

    def save(self, run_dir: Path, competition: CompetitionSpec, profile: DatasetProfile) -> Path:
        brief = self.generate(competition, profile)
        output = run_dir / "brief.md"
        output.write_text(brief)
        logger.info("Saved research brief to %s", output)
        return output

    def _fallback_brief(
        self, competition: CompetitionSpec, profile: DatasetProfile, prompt: str
    ) -> str:
        env_var = _ENV_VAR_BY_PROVIDER.get(self.config.provider.strip().lower(), "OPENAI_API_KEY")
        return (
            render_competition_context(competition)
            + f"# Research Brief: {competition.title or competition.slug}\n\n"
            f"> LLM generation not available. Set {env_var} to enable.\n\n"
            f"## Problem Summary\n\n"
            f"Competition `{competition.slug}` with {profile.row_count} rows and "
            f"{profile.column_count} columns.\n\n"
            f"## Baseline Strategy\n\n"
            f"Start with a LightGBM tabular baseline using {self.config.model} for analysis.\n\n"
            f"## Prompt Preview\n\n"
            f"```\n{prompt[:500]}...\n```\n"
        )
