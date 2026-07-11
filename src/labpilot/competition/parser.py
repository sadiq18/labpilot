import logging
from pathlib import Path
from typing import Protocol

import yaml

from labpilot.competition.metrics import normalize_metric
from labpilot.competition.models import CompetitionMetadata, CompetitionSpec

logger = logging.getLogger(__name__)


class CompetitionMetadataFetcher(Protocol):
    """Anything that can look up a competition's metadata by slug."""

    def fetch_competition_metadata(self, competition: str) -> CompetitionMetadata | None: ...


class CompetitionParser:
    """Fetch and parse a competition's metadata contract.

    A local, per-competition YAML file (`configs/competitions/<slug>.yaml`,
    see its README) always wins when present — it's the only way to override
    things the Kaggle API doesn't expose, like non-standard file-naming
    patterns. When absent, the contract is auto-resolved from the Kaggle API
    (title, description, evaluation metric) via `metadata_fetcher`; anything
    that can't be resolved that way (problem type, file patterns) falls back
    to safe P0 defaults — problem type is inferred later from the profiled
    data, and standard "train*/test*/*submission*" file names are assumed.
    """

    def __init__(
        self,
        competition_slug: str,
        configs_dir: Path | None = None,
        metadata_fetcher: CompetitionMetadataFetcher | None = None,
    ) -> None:
        self.competition_slug = competition_slug
        self.configs_dir = (
            configs_dir or Path(__file__).resolve().parents[3] / "configs" / "competitions"
        )
        self.metadata_fetcher = metadata_fetcher

    def parse(self) -> CompetitionSpec:
        config_path = self.configs_dir / f"{self.competition_slug}.yaml"
        if config_path.is_file():
            spec = self._parse_from_file(config_path)
        else:
            logger.info(
                "No local contract at %s; resolving '%s' automatically.",
                config_path,
                self.competition_slug,
            )
            spec = self._resolve_automatically()

        logger.info(
            "Parsed competition '%s': problem_type=%s, metric=%s",
            self.competition_slug,
            spec.problem_type,
            spec.evaluation_metric.name if spec.evaluation_metric else "unknown",
        )
        return spec

    def _parse_from_file(self, config_path: Path) -> CompetitionSpec:
        logger.info(
            "Parsing competition metadata for '%s' from %s", self.competition_slug, config_path
        )
        raw = yaml.safe_load(config_path.read_text()) or {}
        raw["slug"] = self.competition_slug
        self._apply_default_urls(raw)
        return CompetitionSpec.model_validate(raw)

    def _resolve_automatically(self) -> CompetitionSpec:
        metadata = None
        if self.metadata_fetcher is not None:
            try:
                metadata = self.metadata_fetcher.fetch_competition_metadata(self.competition_slug)
            except Exception:
                logger.warning(
                    "Automatic metadata resolution raised for '%s'; using a bare contract.",
                    self.competition_slug,
                    exc_info=True,
                )

        raw: dict = {"slug": self.competition_slug}
        self._apply_default_urls(raw)
        if metadata is None:
            logger.info(
                "No metadata resolved for '%s'; problem type will be inferred from the "
                "profiled dataset instead.",
                self.competition_slug,
            )
            return CompetitionSpec.model_validate(raw)

        raw["title"] = metadata.title or self.competition_slug
        raw["description"] = metadata.description
        if metadata.category:
            raw["tags"] = [metadata.category]
        metric = normalize_metric(metadata.evaluation_metric_raw)
        if metric is not None:
            raw["evaluation_metric"] = metric.model_dump()
        return CompetitionSpec.model_validate(raw)

    def _apply_default_urls(self, raw: dict) -> None:
        raw.setdefault(
            "data_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/data"
        )
        raw.setdefault(
            "rules_url", f"https://www.kaggle.com/competitions/{self.competition_slug}/rules"
        )

    def save(self, run_dir: Path) -> Path:
        spec = self.parse()
        output = run_dir / "competition.json"
        output.write_text(spec.model_dump_json(indent=2))
        logger.info("Saved competition contract to %s", output)
        return output
