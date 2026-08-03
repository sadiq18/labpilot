"""CompetitionAnalyzer — Kaggle-expert brief (design §3.5 + Plan 5b).

Reuses Milestone 1 ``CompetitionParser`` / ``CompetitionSpec`` as the
fetch/normalize base, then extends into a ``CompetitionProfile``. Overview/rules
pages are enriched via optional ``CompetitionPageAnalyzerAgent`` (LLM) with
deterministic ``rule_engine`` fallback — same typed schema either way.

Winning solutions stay on a capability provider (``NullWinningSolutionProvider``
by default) — no writeup/search scrape in Plan 5.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.competition.models import CompetitionSpec
from labpilot.research_engine.intelligence.competition.page_fetch import CompetitionPages, fetch_competition_pages
from labpilot.research_engine.intelligence.competition.parser import CompetitionMetadataFetcher, CompetitionParser
from labpilot.llm.client import LLMClient
from labpilot.research_engine.intelligence.analyzers.base import BaseAnalyzer
from labpilot.research_engine.intelligence.knowledge import KnowledgeStore
from labpilot.research_engine.intelligence.micro_agents.artifacts import CompetitionPageExtract
from labpilot.research_engine.intelligence.micro_agents.competition_page_analyzer import (
    CompetitionPageAnalyzerAgent,
)
from labpilot.research_engine.intelligence.models import (
    AnalyzeContext,
    ResearchArtifact,
    ResearchArtifacts,
    ResearchArtifactType,
)
from labpilot.research_engine.intelligence.providers.capability import (
    CapabilityResult,
    CompetitionProfile,
    ExternalDataPolicy,
    InferenceLimits,
    RelatedCompetition,
)
from labpilot.research_engine.intelligence.providers.related import (
    RelatedCompetitionProvider,
    SeriesRelatedCompetitionProvider,
)
from labpilot.research_engine.intelligence.providers.winning_solutions import (
    NullWinningSolutionProvider,
    WinningSolutionProvider,
)

logger = logging.getLogger("labpilot.research_engine.intelligence.analyzers.competition")

_RULES_EXCERPT_CAP = 2_000

PageFetcher = Callable[..., CompetitionPages]


class CompetitionAnalyzer(BaseAnalyzer):
    name = "competition"
    default_enabled = True

    def __init__(
        self,
        *,
        metadata_fetcher: CompetitionMetadataFetcher | None = None,
        related_provider: RelatedCompetitionProvider | None = None,
        winning_solution_provider: WinningSolutionProvider | None = None,
        competitions_dir: Path | None = None,
        persist: bool = True,
        llm_client: LLMClient | None = None,
        page_fetcher: PageFetcher | None = None,
        enrich_pages: bool = True,
    ) -> None:
        self.metadata_fetcher = metadata_fetcher
        self.related_provider = related_provider or SeriesRelatedCompetitionProvider(
            metadata_fetcher=metadata_fetcher
        )
        self.winning_solution_provider = (
            winning_solution_provider or NullWinningSolutionProvider()
        )
        self.competitions_dir = competitions_dir
        self.persist = persist
        self.llm_client = llm_client
        self.page_fetcher = page_fetcher or fetch_competition_pages
        self.enrich_pages = enrich_pages
        self._fetcher_explicit = metadata_fetcher is not None
        self._llm_explicit = llm_client is not None

    def analyze(self, context: AnalyzeContext) -> ResearchArtifacts:
        self._maybe_attach_kaggle_client()
        self._maybe_attach_llm_client()
        notes: list[str] = []
        if not self._fetcher_explicit and self.metadata_fetcher is None:
            notes.append(
                "Kaggle credentials not found — using local/YAML contract only "
                "(set KAGGLE_API_TOKEN or ~/.kaggle/access_token for live metadata)."
            )
        spec, source = self._resolve_spec(context, notes)
        profile = self._build_profile(context, spec, notes)
        self._enrich_from_pages(context, profile, notes)

        related_lookup = self.related_provider.find(
            context.competition, context=context, spec=spec
        )
        profile.previous_editions = [
            r for r in related_lookup.related if r.relation == "previous_edition"
        ]
        profile.related_competitions = [
            r for r in related_lookup.related if r.relation != "previous_edition"
        ]
        if related_lookup.capability.status != "ok":
            reason = related_lookup.capability.reason
            suffix = f" — {reason}" if reason else ""
            notes.append(f"related competitions: {related_lookup.capability.status}{suffix}")
            profile.capability_notes.append(reason)

        winning = self.winning_solution_provider.fetch(
            context.competition, context=context
        )
        # Capability provider only — no writeup/search scrape in Plan 5.
        profile.winning_solutions = winning
        notes.append(f"winning solutions: {winning.status} — {winning.reason}")

        profile_artifact = self._profile_artifact(context, profile, source)
        related_artifacts = [
            self._related_artifact(context, related) for related in related_lookup.related
        ]
        items = [profile_artifact, *related_artifacts]

        if self.persist:
            notes.extend(self._persist(context, items))

        return ResearchArtifacts(
            analyzer=self.name,
            items=items,
            notes=notes,
            opportunities=[
                f"related:{r.slug} ({r.relation})" for r in related_lookup.related
            ],
        )

    def _maybe_attach_kaggle_client(self) -> None:
        """Best-effort wire the official Kaggle API when credentials are present."""
        if self._fetcher_explicit or self.metadata_fetcher is not None:
            return
        from labpilot.diagnostics import kaggle_credentials_present

        if not kaggle_credentials_present():
            return
        try:
            from labpilot.config import KaggleConfig, Settings
            from labpilot.accessor.kaggle.client import KaggleClient

            settings = Settings()
            config = KaggleConfig(
                api_token=settings.kaggle_api_token,
                username=settings.kaggle_username,
                key=settings.kaggle_key,
            )
            client = KaggleClient(config)
        except Exception:
            return
        self.metadata_fetcher = client
        if isinstance(self.related_provider, SeriesRelatedCompetitionProvider):
            if self.related_provider.metadata_fetcher is None:
                self.related_provider = SeriesRelatedCompetitionProvider(
                    metadata_fetcher=client
                )

    def _maybe_attach_llm_client(self) -> None:
        """Optional LLM for page enrichment — never required."""
        if self._llm_explicit or self.llm_client is not None:
            return
        try:
            from labpilot.config import Settings, load_config
            from labpilot.llm.client import resolve_llm_client

            config = load_config()
            # Settings may override provider via env; resolve_llm_client handles that.
            _ = Settings()
            self.llm_client = resolve_llm_client(config.llm)
        except Exception:
            self.llm_client = None

    def _resolve_spec(
        self, context: AnalyzeContext, notes: list[str]
    ) -> tuple[CompetitionSpec, str]:
        cached = self._load_cached_spec(context)
        if cached is not None:
            notes.append("Loaded competition contract from local run competition.json.")
            return cached, "m2-cache"

        parser = CompetitionParser(
            context.competition,
            configs_dir=self.competitions_dir,
            metadata_fetcher=self.metadata_fetcher,
            llm_client=None,  # §2.4 — metric normalize is deterministic only
        )
        try:
            spec = parser.parse()
        except Exception as exc:
            notes.append(f"Competition parse failed ({exc}); using bare slug contract.")
            return CompetitionSpec(slug=context.competition), "fallback"
        notes.append("Parsed competition contract via CompetitionParser.")
        return spec, "kaggle" if self.metadata_fetcher is not None else "local"

    def _load_cached_spec(self, context: AnalyzeContext) -> CompetitionSpec | None:
        if not context.runs_dir.is_dir():
            return None
        newest: tuple[str, Path] | None = None
        for run_dir in sorted(context.runs_dir.iterdir()):
            path = run_dir / "competition.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("slug") != context.competition:
                continue
            created = ""
            manifest = run_dir / "manifest.json"
            if manifest.is_file():
                try:
                    created = str(json.loads(manifest.read_text()).get("created_at", ""))
                except (OSError, json.JSONDecodeError):
                    created = ""
            if newest is None or created >= newest[0]:
                newest = (created, path)
        if newest is None:
            return None
        try:
            return CompetitionSpec.model_validate_json(newest[1].read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Could not load cached competition.json: %s", exc)
            return None

    def _build_profile(
        self,
        context: AnalyzeContext,
        spec: CompetitionSpec,
        notes: list[str],
    ) -> CompetitionProfile:
        del notes  # notes filled by enrichment / capabilities
        url = context.url or f"https://www.kaggle.com/competitions/{context.competition}"
        rules_excerpt = (spec.raw_html or "")[:_RULES_EXCERPT_CAP]
        return CompetitionProfile(
            slug=context.competition,
            title=spec.title or context.competition,
            url=url,
            metadata={
                "description": spec.description,
                "tags": list(spec.tags),
                "baseline_strategy": spec.baseline_strategy,
            },
            metric=spec.evaluation_metric,
            problem_type=str(spec.problem_type) if spec.problem_type else None,
            rules_excerpt=rules_excerpt,
            constraints={
                "max_daily_submissions": spec.max_daily_submissions,
                "submissions_disabled": spec.submissions_disabled,
                "is_kernels_submissions_only": spec.is_kernels_submissions_only,
            },
            timeline={
                "deadline": spec.deadline,
                "submissions_disabled": spec.submissions_disabled,
            },
            submission={
                "mode": spec.submission_mode,
                "format": spec.submission_format,
                "columns": list(spec.submission_columns),
                "kernel_output_file": spec.kernel_output_file,
                "submissions_url": spec.submissions_url,
            },
            evaluation={},
            external_data=ExternalDataPolicy(
                status="unavailable",
                notes="Awaiting overview/rules enrichment.",
            ),
            inference_limits=InferenceLimits(
                status="unavailable",
                notes="Awaiting overview/rules enrichment.",
            ),
            dataset_catalog=CapabilityResult(
                available=True,
                status="ok",
                reason="Catalog hints from competition contract (not local EDA).",
            )
            if spec.data_url
            else CapabilityResult(
                available=False,
                status="unavailable",
                reason="No dataset catalog URL on competition contract.",
            ),
            leaderboard=CapabilityResult(
                available=False,
                status="unavailable",
                reason="Leaderboard snapshot not exposed by configured provider.",
            ),
            page_enrichment_source="unavailable",
            capability_notes=[],
        )

    def _enrich_from_pages(
        self,
        context: AnalyzeContext,
        profile: CompetitionProfile,
        notes: list[str],
    ) -> None:
        if not self.enrich_pages:
            notes.append("page enrichment: skipped.")
            return

        try:
            pages = self.page_fetcher(
                context.competition,
                knowledge_dir=context.knowledge_dir,
                refresh=context.refresh,
            )
        except Exception as exc:
            logger.warning("Competition page fetch failed: %s", exc)
            notes.append(f"page enrichment: unavailable — fetch failed ({exc}).")
            profile.capability_notes.append(f"page fetch failed: {exc}")
            return

        if pages.is_empty_shell or not pages.combined_text.strip():
            source = getattr(pages, "source", "none")
            if source == "http":
                reason = (
                    "page content not available without JS render "
                    "(configure Kaggle credentials for the pages API)"
                )
            elif source in {"api", "cache"}:
                reason = "competition pages empty or unusable"
            else:
                reason = (
                    "competition pages unavailable "
                    "(need Kaggle credentials for list_competition_pages)"
                )
            notes.append(f"page enrichment: unavailable — {reason}.")
            profile.external_data = ExternalDataPolicy(status="unavailable", notes=reason)
            profile.inference_limits = InferenceLimits(status="unavailable", notes=reason)
            profile.capability_notes.append(reason)
            # Still keep any rules excerpt we already have for display.
            if pages.rules_text:
                profile.rules_excerpt = pages.rules_text[:_RULES_EXCERPT_CAP]
            return

        if pages.rules_text:
            profile.rules_excerpt = pages.rules_text[:_RULES_EXCERPT_CAP]

        agent = CompetitionPageAnalyzerAgent(llm_client=self.llm_client)
        extract = agent.run(StructuredContext(text=pages.combined_text))
        if not isinstance(extract, CompetitionPageExtract):
            extract = CompetitionPageExtract.model_validate(extract.model_dump())

        # What actually produced this, not what was configured. `uses_llm` is
        # True whenever a client exists, so it recorded "llm" for runs that fell
        # back to the rule engine — provenance that reads as reasoning when the
        # output was deterministic.
        source = agent.last_generated_by
        profile.page_enrichment_source = source
        apply_page_extract(profile, extract)
        notes.append(f"page enrichment: {source}.")
        if profile.external_data.status == "unavailable":
            notes.append(
                f"external data policy: unavailable — {profile.external_data.notes}"
            )
        if profile.inference_limits.status == "unavailable":
            notes.append(
                f"inference limits: unavailable — {profile.inference_limits.notes}"
            )

    def _profile_artifact(
        self, context: AnalyzeContext, profile: CompetitionProfile, source: str
    ) -> ResearchArtifact:
        metric_label = ""
        if profile.metric is not None:
            metric_label = f"{profile.metric.name} ({profile.metric.direction})"
        summary = (
            f"{profile.title} — metric={metric_label or 'unknown'}, "
            f"submission={profile.submission.get('mode', 'unknown')}"
        )
        return ResearchArtifact(
            id=f"competition:{context.competition}",
            type=ResearchArtifactType.COMPETITION,
            source=source,
            title=profile.title,
            summary=summary,
            competition_slug=context.competition,
            confidence=0.9 if source != "fallback" else 0.4,
            metadata={
                "kind": "profile",
                "profile": profile.model_dump(mode="json"),
            },
        )

    def _related_artifact(
        self, context: AnalyzeContext, related: RelatedCompetition
    ) -> ResearchArtifact:
        return ResearchArtifact(
            id=f"competition:{related.slug}",
            type=ResearchArtifactType.COMPETITION,
            source="kaggle",
            title=related.title or related.slug,
            summary=f"{related.relation}: {related.rationale}",
            competition_slug=context.competition,
            confidence=related.score,
            metadata={
                "kind": "related",
                "relation": related.relation,
                "score": related.score,
                "rationale": related.rationale,
                "tags_overlap": related.tags_overlap,
                "related_slug": related.slug,
            },
        )

    def _persist(self, context: AnalyzeContext, items: list[ResearchArtifact]) -> list[str]:
        try:
            with KnowledgeStore(context.knowledge_dir, context.competition) as store:
                for artifact in items:
                    store.upsert_artifact(artifact)
            return [f"Persisted {len(items)} artifact(s) to knowledge.db."]
        except Exception as exc:
            logger.warning("CompetitionAnalyzer persist failed: %s", exc)
            return [f"persist failed: {exc}"]


def apply_page_extract(
    profile: CompetitionProfile, extract: CompetitionPageExtract
) -> None:
    """Merge a typed page extract into the competition profile (deterministic)."""
    if extract.external_data_allowed is None and not extract.external_data_notes:
        profile.external_data = ExternalDataPolicy(
            status="unavailable",
            notes="Could not resolve external-data policy from overview/rules.",
        )
    else:
        profile.external_data = ExternalDataPolicy(
            status="ok" if extract.external_data_allowed is not None else "unavailable",
            allowed=extract.external_data_allowed,
            pretrained_weights=extract.pretrained_weights_allowed,
            notes=extract.external_data_notes
            or (
                "Resolved from overview/rules extract."
                if extract.external_data_allowed is not None
                else "Could not resolve external-data policy from overview/rules."
            ),
        )

    has_inference = any(
        [
            extract.runtime_notes,
            extract.hardware_notes,
            extract.inference_notes,
            extract.internet_allowed is not None,
        ]
    )
    if not has_inference:
        profile.inference_limits = InferenceLimits(
            status="unavailable",
            notes="Inference / kernel resource limits not found in overview/rules.",
        )
    else:
        profile.inference_limits = InferenceLimits(
            status="ok",
            runtime_notes=extract.runtime_notes,
            hardware_notes=extract.hardware_notes,
            internet_allowed=extract.internet_allowed,
            notes=extract.inference_notes or "Resolved from overview/rules extract.",
        )

    profile.evaluation = {
        "formula": extract.evaluation_formula,
        "description": extract.evaluation_description,
    }
    if extract.submission_format:
        profile.submission["format"] = extract.submission_format
    if extract.submission_columns_notes:
        profile.submission["columns_notes"] = extract.submission_columns_notes
    if extract.sample_submission_notes:
        profile.submission["sample_notes"] = extract.sample_submission_notes

    if extract.overview_summary:
        profile.metadata["overview_summary"] = extract.overview_summary
    if extract.other_notes:
        profile.metadata["other_notes"] = extract.other_notes


def profile_dict_for_report(artifact: ResearchArtifact) -> dict[str, Any] | None:
    """Pull a CompetitionProfile dump out of a profile artifact for analyze.json."""
    if artifact.metadata.get("kind") != "profile":
        return None
    profile = artifact.metadata.get("profile")
    return profile if isinstance(profile, dict) else None


def related_dict_for_report(artifact: ResearchArtifact) -> dict[str, Any] | None:
    if artifact.metadata.get("kind") != "related":
        return None
    return {
        "slug": artifact.metadata.get("related_slug")
        or artifact.id.removeprefix("competition:"),
        "title": artifact.title,
        "relation": artifact.metadata.get("relation"),
        "score": artifact.metadata.get("score"),
        "rationale": artifact.metadata.get("rationale", ""),
        "tags_overlap": artifact.metadata.get("tags_overlap", []),
    }
