import logging
from pathlib import Path

import yaml

from labpilot.competition.models import CompetitionSpec

logger = logging.getLogger(__name__)


class CompetitionParser:
    """Fetch and parse a competition's metadata contract.

    P0 reads the contract from a local, per-competition YAML file rather than
    the Kaggle competition page/API (see `configs/competitions/README.md`).
    # TODO: control the verbosity of this class's logging via a future CLI
    # --verbose/--quiet flag (see docs/MILESTONES.md).
    """

    def __init__(self, competition_slug: str, configs_dir: Path | None = None) -> None:
        self.competition_slug = competition_slug
        self.configs_dir = (
            configs_dir or Path(__file__).resolve().parents[3] / "configs" / "competitions"
        )

    def parse(self) -> CompetitionSpec:
        config_path = self.configs_dir / f"{self.competition_slug}.yaml"
        logger.info(
            "Parsing competition metadata for '%s' from %s", self.competition_slug, config_path
        )
        if not config_path.is_file():
            raise ValueError(
                f"Competition '{self.competition_slug}' is not supported in P0. "
                f"Expected metadata at {config_path}. See "
                f"{config_path.parent / 'README.md'} for the expected schema."
            )

        raw = yaml.safe_load(config_path.read_text()) or {}
        raw["slug"] = self.competition_slug
        raw.setdefault(
            "data_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/data"
        )
        raw.setdefault(
            "rules_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/rules"
        )
        spec = CompetitionSpec.model_validate(raw)
        logger.info(
            "Parsed competition '%s': problem_type=%s, metric=%s",
            self.competition_slug,
            spec.problem_type,
            spec.evaluation_metric.name if spec.evaluation_metric else "unknown",
        )
        return spec

    def save(self, run_dir: Path) -> Path:
        spec = self.parse()
        output = run_dir / "competition.json"
        output.write_text(spec.model_dump_json(indent=2))
        logger.info("Saved competition contract to %s", output)
        return output
