"""Workspace capability — create/verify competition code/data layout.

Capability ``name`` stays ``\"workspace\"``. The on-disk root is the competition
slug directory (client workspace or legacy ``competitions/<slug>/``).

Also prepares data + profile when possible (scaffold ``research init`` does not
download): download via accessor (when not dry-run / not skipped), write
``profile.json``, and persist ``competition.json`` from Intelligence parser
artifacts or a local contract (``configs/competition.yaml`` in a workspace).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from labpilot.research_engine.execution.capabilities.base import BaseCapability
from labpilot.research_engine.execution.context import TaskContext
from labpilot.research_engine.execution.schemas import TaskEvidence
from labpilot.research_engine.planner.schemas.task_types import TaskType

logger = logging.getLogger(__name__)


def _cache_may_serve(kaggle: object, context: TaskContext) -> bool:
    """Is there a download cache that could satisfy this competition offline?

    `DataDownloader.download` reads `cache_dir` before authenticating, so a
    warm cache is a legitimate way to have data without credentials — and the
    first version of this gate refused before ever trying. Reported on PR #120.
    """
    cache_dir = getattr(kaggle, "cache_dir", None)
    if cache_dir is None:
        return False
    cached = Path(cache_dir).expanduser()
    if not cached.is_absolute():
        cached = Path(context.workspace_root) / cached
    return cached.is_dir() and any(cached.rglob("*"))


def _has_credentials(kaggle: object) -> bool:
    """Are there credentials here that could actually authenticate?

    `kaggle is None` was the test, and it asked whether a *constraint had been
    passed* rather than whether credentials exist — the M20 shape, inside the
    M20 change. It got the diagnosis wrong in both directions: a workspace with
    no `configs/default.yaml` failed with *"no Kaggle credentials"* when the
    real cause was no config at all, and a config carrying an **empty**
    `KaggleConfig` sailed through the check and failed further down, where the
    message is about the download rather than the setup. Reported on PR #120,
    four times, and right every time. The tests kept missing it because they ran
    against a workspace that has a config.
    """
    if kaggle is None:
        return False
    token = str(getattr(kaggle, "api_token", "") or "").strip()
    username = str(getattr(kaggle, "username", "") or "").strip()
    key = str(getattr(kaggle, "key", "") or "").strip()
    return bool(token or (username and key))

#: Relative dirs under the competition workspace root (idempotent).
_WORKSPACE_SUBDIRS = (
    "pipeline",
    "src",
    "configs",
    "data",
    "data/raw",
    "data/processed",
    "logs",
    "artifacts",
    "models",
    "tests",
)


class WorkspaceCapability(BaseCapability):
    name = "workspace"

    @property
    def supported_task_types(self) -> frozenset[TaskType]:
        return frozenset({TaskType.PREPARE_WORKSPACE})

    def execute(self, context: TaskContext) -> TaskEvidence:
        root = context.workspace_root
        expected_name = context.competition
        root.mkdir(parents=True, exist_ok=True)
        created: list[str] = []
        for name in _WORKSPACE_SUBDIRS:
            path = root / name
            existed = path.is_dir()
            path.mkdir(parents=True, exist_ok=True)
            if not existed:
                created.append(str(path))

        context.paths.ensure()
        research_ok = context.paths.root.is_dir()
        named_ok = root.name == expected_name

        checks = ["dirs_exist", "writable", "named_as_competition"]
        metadata: dict[str, Any] = {
            "created": created,
            "idempotent": not created,
            "workspace": str(root),
            "competition": expected_name,
        }
        errors: list[str] = []
        if not named_ok:
            errors.append(f"workspace must be named {expected_name!r}, got {root.name!r}")

        competition_path = self._ensure_competition_json(context, root, metadata, errors)
        if competition_path is not None:
            checks.append("competition_json")

        download_ok = self._ensure_data(context, root, metadata, errors, checks)
        profile_ok = self._ensure_profile(context, root, metadata, errors, checks)

        passed = (
            research_ok
            and root.is_dir()
            and named_ok
            and (download_ok is not False)
            and (profile_ok is not False)
            and not errors
        )
        # Soft-fail download/profile in dry-run — dirs alone are enough to proceed.
        if context.constraints.get("dry_run") and named_ok and research_ok:
            passed = True
            errors = [e for e in errors if "workspace must be named" in e]

        summary = "workspace prepared" if created else "workspace already present"
        if metadata.get("downloaded"):
            summary = "workspace prepared (data downloaded)"
        elif metadata.get("data_reused"):
            summary = "workspace prepared (cached data)"
        elif not passed:
            # The card has to agree with the verdict. It read *"workspace
            # prepared; download skipped"* beside `passed=False` — the summary
            # describing the old, silent behaviour while the verdict described
            # the new one, which is a card that argues with itself. Reported on
            # PR #120.
            summary = "workspace not prepared: " + (errors[0] if errors else "see error")
        elif metadata.get("download_skipped"):
            summary = f"{summary}; download skipped"

        return TaskEvidence(
            task_id=context.task.id,
            execution_id=context.execution.id,
            capability=self.name,
            passed=passed,
            summary=summary,
            checks=checks,
            paths=[str(root / name) for name in _WORKSPACE_SUBDIRS],
            error="; ".join(errors) if errors else None,
            metadata=metadata,
        )

    def _ensure_competition_json(
        self,
        context: TaskContext,
        root: Path,
        metadata: dict[str, Any],
        errors: list[str],
    ) -> Path | None:
        out = root / "competition.json"
        if out.is_file():
            metadata["competition_json"] = str(out)
            metadata["competition_json_reused"] = True
            return out

        # Prefer Analyze / knowledge copy if present.
        knowledge_candidate = context.paths.root / "competition.json"
        if knowledge_candidate.is_file():
            out.write_text(knowledge_candidate.read_text(encoding="utf-8"), encoding="utf-8")
            metadata["competition_json"] = str(out)
            metadata["competition_json_source"] = "knowledge"
            return out

        # Hydrate from Analyze report when available (tags / description / metric).
        analyze_path = context.paths.report_path
        if analyze_path.is_file():
            try:
                import json as _json

                from labpilot.research_engine.intelligence.competition.infer_problem_type import (
                    infer_problem_type_from_metadata,
                )
                from labpilot.research_engine.intelligence.competition.models import (
                    CompetitionSpec,
                    MetricSpec,
                    ProblemType,
                )

                report = _json.loads(analyze_path.read_text(encoding="utf-8"))
                comp = report.get("competition") or {}
                meta = (comp.get("metadata") or {}) if isinstance(comp, dict) else {}
                nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else meta
                tags = list(nested.get("tags") or meta.get("tags") or [])
                description = str(
                    nested.get("description")
                    or nested.get("overview_summary")
                    or comp.get("description")
                    or ""
                )
                title = str(comp.get("title") or context.competition)
                metric_raw = comp.get("metric") or {}
                metric = None
                if isinstance(metric_raw, dict) and metric_raw.get("name"):
                    metric = MetricSpec(
                        name=str(metric_raw.get("name")),
                        direction=str(metric_raw.get("direction") or "maximize"),
                        description=str(metric_raw.get("description") or ""),
                        key=metric_raw.get("key"),
                    )
                problem_type = ProblemType.UNKNOWN
                raw_pt = str(comp.get("problem_type") or "unknown")
                try:
                    problem_type = ProblemType(raw_pt)
                except ValueError:
                    problem_type = ProblemType.UNKNOWN
                if problem_type is ProblemType.UNKNOWN:
                    problem_type = infer_problem_type_from_metadata(
                        title=title,
                        description=description,
                        tags=tags,
                        metric_name=(metric.name if metric else ""),
                        metric_description=(metric.description if metric else ""),
                    )
                spec = CompetitionSpec(
                    slug=context.competition,
                    title=title,
                    description=description,
                    tags=tags,
                    problem_type=problem_type,
                    evaluation_metric=metric,
                )
                out.write_text(spec.model_dump_json(indent=2) + "\n", encoding="utf-8")
                metadata["competition_json"] = str(out)
                metadata["competition_json_source"] = "analyze"
                return out
            except Exception as exc:
                logger.info("Could not hydrate competition.json from analyze: %s", exc)

        configs_dir = context.constraints.get("competitions_dir")
        if configs_dir is None:
            # Client workspace: prefer local configs/ next to labpilot.yaml
            local_configs = root / "configs"
            if (local_configs / "competition.yaml").is_file() or (
                local_configs / f"{context.competition}.yaml"
            ).is_file():
                configs_dir = local_configs
        try:
            from labpilot.research_engine.intelligence.competition.parser import (
                CompetitionParser,
            )

            parser = CompetitionParser(
                context.competition,
                Path(configs_dir) if configs_dir else None,
            )
            parser.save(root)
            metadata["competition_json"] = str(out)
            metadata["competition_json_source"] = "parser"
            return out
        except Exception as exc:
            logger.info("competition.json not written: %s", exc)
            metadata["competition_json_skipped"] = str(exc)
            # Non-fatal — Code Engineering has defaults.
            return None

    def _ensure_data(
        self,
        context: TaskContext,
        root: Path,
        metadata: dict[str, Any],
        errors: list[str],
        checks: list[str],
    ) -> bool | None:
        """Return True if data ready, False on hard failure, None if skipped."""
        raw_dir = root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in raw_dir.rglob("*") if p.is_file()]
        if existing:
            metadata["data_reused"] = True
            metadata["raw_file_count"] = len(existing)
            checks.append("data_present")
            return True

        skip = bool(
            context.constraints.get("skip_download")
            or context.constraints.get("dry_run")
            or context.constraints.get("skip_data_download")
        )
        if skip:
            metadata["download_skipped"] = True
            checks.append("download_skipped")
            return None

        kaggle = context.constraints.get("kaggle")
        client = context.constraints.get("kaggle_client")
        # Asked only when a download is the *only* way to get data. `DataDownloader`
        # serves from `cache_dir` before it authenticates, so a workspace with a
        # warm shared cache used to succeed offline — and refusing here on
        # credentials alone took that away without ever trying. Reported on
        # PR #120. The verdict below is about whether data arrived; credentials
        # are a diagnosis, not the gate.
        unusable = client is None and not _has_credentials(kaggle)
        if unusable and not _cache_may_serve(kaggle, context):
            # **Skipped because asked to** and **skipped because unable** were
            # both `None`, and the verdict read anything-but-False as done. So a
            # workspace with no credentials reported `passed=True` carrying
            # `download_skipped: no_kaggle_config` in its own metadata: it said
            # what was wrong and passed anyway, and every step after it ran
            # against an empty tree. M20 finding, 2026-08-09.
            metadata["download_skipped"] = "no_kaggle_credentials"
            checks.append("download_unavailable")
            errors.append(
                "no usable Kaggle credentials and nothing cached, so the dataset was "
                "never fetched. Set KAGGLE_API_TOKEN (or KAGGLE_USERNAME and "
                "KAGGLE_KEY) in the workspace `.env`, or pass --dry-run if this run "
                "does not need data."
            )
            return False

        try:
            from labpilot.accessor.data.downloader import DataDownloader
            from labpilot.config import KaggleConfig

            config = (
                kaggle if isinstance(kaggle, KaggleConfig) else KaggleConfig.model_validate(kaggle)
            )
            downloader = DataDownloader(context.competition, config, client=client)
            downloader.download(root)
            files = [p for p in raw_dir.rglob("*") if p.is_file()]
            metadata["downloaded"] = True
            metadata["raw_file_count"] = len(files)
            checks.append("data_downloaded")
            return bool(files)
        except Exception as exc:
            logger.warning("Workspace data download failed: %s", exc)
            metadata["download_error"] = str(exc)
            errors.append(f"data download failed: {exc}")
            return False

    def _ensure_profile(
        self,
        context: TaskContext,
        root: Path,
        metadata: dict[str, Any],
        errors: list[str],
        checks: list[str],
    ) -> bool | None:
        profile_path = root / "profile.json"
        if profile_path.is_file():
            metadata["profile_reused"] = True
            metadata["profile"] = str(profile_path)
            checks.append("profile_present")
            return True

        raw_dir = root / "data" / "raw"
        raw_files = [p for p in raw_dir.rglob("*") if p.is_file()] if raw_dir.is_dir() else []
        if not raw_files:
            if context.constraints.get("dry_run"):
                metadata["profile_skipped"] = "no_data_dry_run"
                checks.append("profile_skipped")
                return None
            metadata["profile_skipped"] = "no_data"
            checks.append("profile_skipped")
            return None

        try:
            from labpilot.accessor.profiler.report import write_profile
            from labpilot.accessor.profiler.tabular import TabularProfiler
            from labpilot.config import ProfilerConfig
            from labpilot.research_engine.intelligence.competition.models import (
                CompetitionSpec,
            )

            profiler_cfg = context.constraints.get("profiler")
            config = (
                profiler_cfg
                if isinstance(profiler_cfg, ProfilerConfig)
                else ProfilerConfig.model_validate(profiler_cfg or {})
            )
            competition = CompetitionSpec(slug=context.competition)
            comp_path = root / "competition.json"
            if comp_path.is_file():
                competition = CompetitionSpec.model_validate_json(
                    comp_path.read_text(encoding="utf-8")
                )

            profile = TabularProfiler(config).profile_directory(
                raw_dir,
                context.competition,
                train_pattern=competition.train_file_pattern,
                test_pattern=competition.test_file_pattern,
                submission_pattern=competition.submission_file_pattern,
                competition_title=competition.title,
                competition_description=competition.description,
            )
            json_path, md_path = write_profile(root, profile)
            metadata["profile"] = str(json_path)
            metadata["profile_md"] = str(md_path)
            checks.append("profile_written")
            return True
        except Exception as exc:
            logger.warning("Workspace tabular profile failed: %s", exc)
            metadata["profile_error"] = str(exc)
            # Non-CSV / scientific layouts: still write an inventory profile so
            # baseline selection + codegen know what files exist.
            inventory_ok = self._write_inventory_profile(
                context, root, raw_dir, metadata, checks, str(exc)
            )
            if inventory_ok:
                return True
            checks.append("profile_skipped")
            return None

    def _write_inventory_profile(
        self,
        context: TaskContext,
        root: Path,
        raw_dir: Path,
        metadata: dict[str, Any],
        checks: list[str],
        tabular_error: str,
    ) -> bool:
        try:
            from labpilot.accessor.profiler.modality import IMAGE_EXTENSIONS, ModalityDetector
            from labpilot.accessor.profiler.report import write_profile
            from labpilot.accessor.profiler.tabular import DatasetProfile
            from labpilot.research_engine.intelligence.competition.infer_problem_type import (
                infer_problem_type_from_metadata,
            )
            from labpilot.research_engine.intelligence.competition.models import (
                CompetitionSpec,
                ProblemType,
            )

            competition = CompetitionSpec(slug=context.competition)
            comp_path = root / "competition.json"
            if comp_path.is_file():
                competition = CompetitionSpec.model_validate_json(
                    comp_path.read_text(encoding="utf-8")
                )

            files: list[str] = []
            for path in sorted(raw_dir.rglob("*")):
                if path.is_file():
                    files.append(str(path.relative_to(raw_dir)))
                if len(files) >= 200:
                    break

            profile = DatasetProfile(
                competition=context.competition,
                files=files,
                warnings=[
                    f"tabular profiler unavailable: {tabular_error}",
                    "using filesystem inventory profile",
                ],
            )
            modality = ModalityDetector().detect(
                raw_dir,
                profile,
                competition_title=competition.title,
                competition_description=competition.description,
            )
            if modality.modality == "tabular":
                # Metadata may still say vision/tracking when files are exotic.
                inferred = infer_problem_type_from_metadata(
                    title=competition.title,
                    description=competition.description,
                    tags=list(competition.tags),
                )
                if inferred is ProblemType.IMAGE_CLASSIFICATION:
                    profile.modality = "image"
                    profile.warnings.append("modality=image from competition metadata")
                elif inferred is ProblemType.TEXT_CLASSIFICATION:
                    profile.modality = "text"
                    profile.warnings.append("modality=text from competition metadata")
                else:
                    # Presence of image-like files without CSV roles.
                    has_images = any(
                        Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in files
                    )
                    has_zarr = any(".zarr" in f for f in files) or any(
                        p.is_dir() and p.name.endswith(".zarr") for p in raw_dir.rglob("*")
                    )
                    if has_images or has_zarr:
                        profile.modality = "image"
                        profile.warnings.append("modality=image from file extensions")
                    else:
                        profile.modality = modality.modality
            else:
                profile.modality = modality.modality
            profile.image_dir = modality.image_dir
            profile.image_column = modality.image_column
            profile.text_column = modality.text_column
            if modality.signals:
                profile.warnings.extend(modality.signals)

            json_path, md_path = write_profile(root, profile)
            metadata["profile"] = str(json_path)
            metadata["profile_md"] = str(md_path)
            metadata["profile_kind"] = "inventory"
            checks.append("profile_inventory")
            return True
        except Exception as inv_exc:
            logger.warning("Inventory profile failed: %s", inv_exc)
            metadata["inventory_profile_error"] = str(inv_exc)
            return False


def default_workspace_dirs(root: Path) -> list[Path]:
    return [
        root / name
        for name in ("pipeline", "src", "configs", "data", "logs", "artifacts", "tests")
    ]
