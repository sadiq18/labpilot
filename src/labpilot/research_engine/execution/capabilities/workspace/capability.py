"""Workspace capability — create/verify ``competitions/<slug>/`` layout.

Capability ``name`` stays ``\"workspace\"``. The on-disk root is the competition
slug directory (not the execution id).

Also prepares data + profile when possible so ``research init`` is unnecessary:
download via accessor (when not dry-run / not skipped), write ``profile.json``,
and persist ``competition.json`` from Intelligence parser artifacts or a local
contract.
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

#: Relative dirs created under ``competitions/<competition-slug>/`` (idempotent).
_WORKSPACE_SUBDIRS = (
    "src",
    "configs",
    "data",
    "data/raw",
    "data/processed",
    "logs",
    "artifacts",
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

        configs_dir = context.constraints.get("competitions_dir")
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
        if kaggle is None and client is None:
            metadata["download_skipped"] = "no_kaggle_config"
            checks.append("download_skipped")
            return None

        try:
            from labpilot.accessor.data.downloader import DataDownloader
            from labpilot.config import KaggleConfig

            config = kaggle if isinstance(kaggle, KaggleConfig) else KaggleConfig.model_validate(kaggle)
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
            from labpilot.config import ProfilerConfig
            from labpilot.accessor.profiler.report import write_profile
            from labpilot.accessor.profiler.tabular import TabularProfiler
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
            logger.warning("Workspace profile failed: %s", exc)
            metadata["profile_error"] = str(exc)
            checks.append("profile_skipped")
            # Non-fatal — Code Engineering falls back to defaults without a profile.
            return None


def default_workspace_dirs(root: Path) -> list[Path]:
    return [root / name for name in ("src", "configs", "data", "logs", "artifacts", "tests")]
