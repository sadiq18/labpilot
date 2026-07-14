"""Standalone HTML report generation for completed research runs."""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from labpilot.baseline.selector import BaselineChoice
from labpilot.competition.models import CompetitionSpec
from labpilot.kaggle.client import SubmissionResult
from labpilot.orchestrator.manifest import RunManifest, load_manifest
from labpilot.profiler.report import load_profile
from labpilot.runtimes.models import RuntimeRecord

logger = logging.getLogger(__name__)


def markdown_to_html(text: str) -> str:
    """Convert Markdown to HTML, with a small stdlib fallback when markdown is unavailable."""
    if not text.strip():
        return "<p><em>No content.</em></p>"
    try:
        import markdown

        return markdown.markdown(
            text,
            extensions=["extra", "sane_lists", "tables", "fenced_code"],
        )
    except ImportError:
        return _fallback_markdown_to_html(text)


def _fallback_markdown_to_html(text: str) -> str:
    lines = text.splitlines()
    parts: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                parts.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("#"):
            if in_list:
                parts.append("</ul>")
                in_list = False
            level = len(stripped) - len(stripped.lstrip("#"))
            title = html.escape(stripped[level:].strip())
            parts.append(f"<h{min(level, 6)}>{title}</h{min(level, 6)}>")
            continue
        if stripped.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{html.escape(stripped[2:].strip())}</li>")
            continue
        if in_list:
            parts.append("</ul>")
            in_list = False
        parts.append(f"<p>{html.escape(stripped)}</p>")
    if in_list:
        parts.append("</ul>")
    return "\n".join(parts)


class ReportGenerator:
    """Render a self-contained HTML report from run artifacts."""

    def __init__(self) -> None:
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            auto_reload=False,
        )

    def generate(self, run_dir: Path, manifest: RunManifest | None = None) -> Path:
        run_dir = run_dir.resolve()
        manifest = manifest or load_manifest(run_dir)
        context = self.build_context(run_dir, manifest)
        rendered = self.env.get_template("report.html.j2").render(**context)
        output = run_dir / "report.html"
        output.write_text(rendered, encoding="utf-8")
        logger.info("Saved HTML report to %s", output)
        return output

    def build_context(self, run_dir: Path, manifest: RunManifest) -> dict[str, Any]:
        competition = self._load_json_model(run_dir / "competition.json", CompetitionSpec)
        profile = load_profile(run_dir)
        baseline = self._load_json_model(run_dir / "baseline_choice.json", BaselineChoice)
        submission = self._load_submission(run_dir)
        runtime = self._load_runtime(run_dir)
        metrics = self._load_metrics(run_dir)

        brief_html = markdown_to_html(self._read_text(run_dir / "brief.md"))
        reflection_html = markdown_to_html(self._read_text(run_dir / "reflection.md"))
        profile_html = markdown_to_html(self._read_text(run_dir / "profile.md"))

        # Relative link to competition dashboard when generated (Plan 8).
        dashboard_href = None
        dash_path = (
            run_dir.resolve().parent.parent
            / "knowledge"
            / manifest.competition
            / "dashboard.html"
        )
        if dash_path.is_file():
            dashboard_href = f"../../knowledge/{manifest.competition}/dashboard.html"

        return {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "run_id": manifest.run_id,
            "competition_slug": manifest.competition,
            "manifest_status": manifest.status.value,
            "competition": competition,
            "profile": profile,
            "baseline": baseline,
            "metrics": metrics,
            "submission": submission,
            "runtime": runtime,
            "brief_html": brief_html,
            "reflection_html": reflection_html,
            "profile_html": profile_html,
            "stages": self._stage_rows(manifest),
            "lineage": self._lineage(manifest),
            "dashboard_href": dashboard_href,
        }

    @staticmethod
    def _read_text(path: Path) -> str:
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _load_json_model(path: Path, model_cls: type[Any]) -> Any | None:
        if not path.is_file():
            return None
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_metrics(run_dir: Path) -> dict[str, float]:
        path = run_dir / "metrics.json"
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {key: float(value) for key, value in raw.items() if isinstance(value, (int, float))}

    @staticmethod
    def _load_submission(run_dir: Path) -> SubmissionResult | None:
        path = run_dir / "submission_result.json"
        if not path.is_file():
            return None
        return SubmissionResult.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_runtime(run_dir: Path) -> RuntimeRecord | None:
        path = run_dir / "runtime.json"
        if not path.is_file():
            return None
        return RuntimeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    @staticmethod
    def _stage_rows(manifest: RunManifest) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for stage in manifest.stages:
            rows.append(
                {
                    "name": stage.name,
                    "status": stage.status.value,
                    "started_at": str(stage.started_at or ""),
                    "finished_at": str(stage.finished_at or ""),
                    "error": stage.error or "",
                }
            )
        return rows

    @staticmethod
    def _lineage(manifest: RunManifest) -> dict[str, Any]:
        metadata = manifest.metadata or {}
        if not metadata.get("parent_run_id"):
            return {}
        return {
            "parent_run_id": metadata.get("parent_run_id"),
            "iteration": metadata.get("iteration"),
            "improvement_strategy": metadata.get("improvement_strategy"),
        }
