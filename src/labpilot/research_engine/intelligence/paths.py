"""Canonical on-disk layout for a competition's research tree (knowledge-system.md §4).

``raw/`` ≠ ``extracted/`` ≠ ``knowledge/`` — different directories and jobs. This
module is the single source of truth for those paths; both ``AnalyzeContext``
and the ``KnowledgeStore`` build on it so they can never drift apart.

Everything here lives under ``knowledge/<slug>/research/`` and is **local only /
gitignored** — never committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAW_SUBDIRS = ("papers", "repositories", "kernels", "discussions", "competitions")
EXTRACTED_SUBDIRS = ("papers", "repositories", "forums")
KNOWLEDGE_SUBDIRS = ("techniques", "datasets", "architectures", "tasks")


@dataclass(frozen=True)
class ResearchPaths:
    """Resolve the research tree for one competition.

    ``base_dir`` is the repo-level knowledge directory (``config.knowledge_dir``,
    e.g. ``knowledge/``); ``competition`` is the normalized slug.
    """

    base_dir: Path
    competition: str

    @property
    def root(self) -> Path:
        return self.base_dir / self.competition / "research"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.root / "extracted"

    @property
    def knowledge_dir(self) -> Path:
        """Layer 3 — merged knowledge objects (``research/knowledge/``)."""
        return self.root / "knowledge"

    @property
    def experiments_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def embeddings_dir(self) -> Path:
        """Future optional (Stage 3) — created empty in M3 v1."""
        return self.root / "embeddings"

    @property
    def db_path(self) -> Path:
        return self.root / "knowledge.db"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "analyze.json"

    @property
    def brief_path(self) -> Path:
        """Durable Research Brief markdown written by ``research analyze``."""
        return self.reports_dir / "research_brief.md"

    def all_dirs(self) -> list[Path]:
        dirs = [
            self.raw_dir,
            self.extracted_dir,
            self.knowledge_dir,
            self.experiments_dir,
            self.reports_dir,
            self.embeddings_dir,
        ]
        dirs += [self.raw_dir / sub for sub in RAW_SUBDIRS]
        dirs += [self.extracted_dir / sub for sub in EXTRACTED_SUBDIRS]
        dirs += [self.knowledge_dir / sub for sub in KNOWLEDGE_SUBDIRS]
        return dirs

    def ensure(self) -> ResearchPaths:
        """Create the full locked tree (idempotent) and return self."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self
