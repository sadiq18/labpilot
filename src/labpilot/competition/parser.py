from pathlib import Path

import yaml

from labpilot.competition.models import CompetitionSpec


class CompetitionParser:
    """Fetch and parse Kaggle competition metadata."""

    def __init__(self, competition_slug: str, configs_dir: Path | None = None) -> None:
        self.competition_slug = competition_slug
        self.configs_dir = (
            configs_dir or Path(__file__).resolve().parents[3] / "configs" / "competitions"
        )

    def parse(self) -> CompetitionSpec:
        config_path = self.configs_dir / f"{self.competition_slug}.yaml"
        if not config_path.is_file():
            raise ValueError(
                f"Competition '{self.competition_slug}' is not supported in P0. "
                f"Expected metadata at {config_path}."
            )

        raw = yaml.safe_load(config_path.read_text()) or {}
        raw["slug"] = self.competition_slug
        raw.setdefault(
            "data_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/data"
        )
        raw.setdefault(
            "rules_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/rules"
        )
        return CompetitionSpec.model_validate(raw)

    def save(self, run_dir: Path) -> Path:
        spec = self.parse()
        output = run_dir / "competition.json"
        output.write_text(spec.model_dump_json(indent=2))
        return output
