from pathlib import Path

from labpilot.competition.models import CompetitionSpec, MetricSpec, ProblemType


class CompetitionParser:
    """Fetch and parse Kaggle competition metadata."""

    def __init__(self, competition_slug: str) -> None:
        self.competition_slug = competition_slug

    def parse(self) -> CompetitionSpec:
        # TODO: integrate Kaggle API + page scrape for full metadata
        return CompetitionSpec(
            slug=self.competition_slug,
            title=self.competition_slug.replace("-", " ").title(),
            description="Competition description will be fetched from Kaggle.",
            evaluation_metric=MetricSpec(name="unknown", direction="maximize"),
            problem_type=ProblemType.UNKNOWN,
            data_url=f"https://www.kaggle.com/competitions/{self.competition_slug}/data",
            rules_url=f"https://www.kaggle.com/competitions/{self.competition_slug}/rules",
        )

    def save(self, run_dir: Path) -> Path:
        spec = self.parse()
        output = run_dir / "competition.json"
        output.write_text(spec.model_dump_json(indent=2))
        return output
