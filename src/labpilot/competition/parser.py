import logging
from datetime import datetime
from pathlib import Path
from typing import Protocol

import yaml

from labpilot.competition.metrics import enrich_metric_spec, normalize_metric
from labpilot.competition.models import CompetitionMetadata, CompetitionSpec
from labpilot.competition.rules import fetch_rules_excerpt
from labpilot.llm.client import LLMClient

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
        llm_client: LLMClient | None = None,
    ) -> None:
        self.competition_slug = competition_slug
        self.configs_dir = (
            configs_dir or Path(__file__).resolve().parents[3] / "configs" / "competitions"
        )
        self.metadata_fetcher = metadata_fetcher
        self.llm_client = llm_client

    def parse(self) -> CompetitionSpec:
        config_path = self.configs_dir / f"{self.competition_slug}.yaml"
        spec = self._parse_from_file(config_path) if config_path.is_file() else self._resolve_automatically()
        spec = self._apply_rules_scrape(spec)
        self._warn_if_competition_closed(spec)

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
        spec = CompetitionSpec.model_validate(raw)
        return self._enrich_metric(spec)

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
        if metadata.tags:
            raw["tags"] = metadata.tags
        elif metadata.category:
            raw["tags"] = [metadata.category]
        if metadata.deadline:
            raw["deadline"] = metadata.deadline
        if metadata.max_daily_submissions is not None:
            raw["max_daily_submissions"] = metadata.max_daily_submissions
        raw["submissions_disabled"] = metadata.submissions_disabled
        raw["is_kernels_submissions_only"] = metadata.is_kernels_submissions_only
        metric = normalize_metric(metadata.evaluation_metric_raw)
        if metric is not None:
            metric = enrich_metric_spec(
                metric,
                metadata.evaluation_metric_raw,
                llm_client=self.llm_client,
            )
            raw["evaluation_metric"] = metric.model_dump()
        spec = CompetitionSpec.model_validate(raw)
        return self._enrich_metric(spec)

    def _enrich_metric(self, spec: CompetitionSpec) -> CompetitionSpec:
        """Fill in metric.key via rules or LLM when missing from local YAML."""
        if spec.evaluation_metric is None:
            return spec
        metric = spec.evaluation_metric
        if metric.key is not None:
            return spec
        raw_text = metric.description or metric.name
        enriched = enrich_metric_spec(metric, raw_text, llm_client=self.llm_client)
        if enriched.key == metric.key:
            return spec
        return spec.model_copy(update={"evaluation_metric": enriched})

    def _apply_rules_scrape(self, spec: CompetitionSpec) -> CompetitionSpec:
        if spec.raw_html:
            return spec
        excerpt = fetch_rules_excerpt(spec.rules_url)
        if not excerpt:
            return spec
        return spec.model_copy(update={"raw_html": excerpt})

    def _warn_if_competition_closed(self, spec: CompetitionSpec) -> None:
        if spec.submissions_disabled:
            logger.warning(
                "Competition '%s' has submissions disabled on Kaggle.",
                self.competition_slug,
            )
        if spec.deadline:
            try:
                deadline = datetime.fromisoformat(spec.deadline.replace("Z", "+00:00"))
                if deadline.tzinfo is not None:
                    deadline = deadline.replace(tzinfo=None)
                if deadline < datetime.now():
                    logger.warning(
                        "Competition '%s' deadline (%s) has already passed.",
                        self.competition_slug,
                        spec.deadline,
                    )
            except ValueError:
                pass

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
