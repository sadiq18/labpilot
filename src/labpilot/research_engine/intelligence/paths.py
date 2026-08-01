"""Canonical on-disk layout for a competition's research tree (knowledge-system.md §4).

``raw/`` ≠ ``extracted/`` ≠ ``knowledge/`` — different directories and jobs. This
module is the single source of truth for those paths; both ``AnalyzeContext``
and the ``KnowledgeStore`` build on it so they can never drift apart.

**Client workspace** (``labpilot.yaml``): ``<ws>/knowledge/research/…`` (flat).  
**Legacy multi-slug:** ``knowledge/<slug>/research/…``.

Competition-local only / gitignored — never commit ``knowledge.db``. Shared
transferable memory lives in ``experiences.db`` outside the workspace
(:func:`labpilot.workspace.resolve_experience_db_path`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from labpilot.workspace import (
    competition_data_root,
    is_client_knowledge_layout,
    migrate_nested_client_knowledge,
)

RAW_SUBDIRS = ("papers", "repositories", "kernels", "discussions", "competitions")
EXTRACTED_SUBDIRS = ("papers", "repositories", "forums")
KNOWLEDGE_SUBDIRS = ("techniques", "datasets", "architectures", "tasks")


@dataclass(frozen=True)
class ResearchPaths:
    """Resolve the research tree for one competition.

    ``base_dir`` is the knowledge directory (``config.knowledge_dir``, e.g.
    ``<ws>/knowledge`` or legacy ``knowledge/``); ``competition`` is the slug.
    """

    base_dir: Path
    competition: str

    @property
    def data_root(self) -> Path:
        """Competition data root (flat ``knowledge/`` or legacy ``knowledge/<slug>``)."""
        return competition_data_root(self.base_dir, self.competition)

    @property
    def is_client_layout(self) -> bool:
        return is_client_knowledge_layout(self.base_dir, self.competition)

    @property
    def root(self) -> Path:
        return self.data_root / "research"

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
    def plans_dir(self) -> Path:
        """Research Planner projections (``<plan_id>.json`` / ``.md``)."""
        return self.root / "plans"

    @property
    def executions_dir(self) -> Path:
        """Research Engineer execution workspaces (``E-xxx/``)."""
        return self.root / "executions"

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
            self.plans_dir,
            self.executions_dir,
            self.reports_dir,
            self.embeddings_dir,
        ]
        dirs += [self.raw_dir / sub for sub in RAW_SUBDIRS]
        dirs += [self.extracted_dir / sub for sub in EXTRACTED_SUBDIRS]
        dirs += [self.knowledge_dir / sub for sub in KNOWLEDGE_SUBDIRS]
        return dirs

    def ensure(self) -> ResearchPaths:
        """Migrate nested client layout if needed, create tree, return self."""
        migrate_nested_client_knowledge(self.base_dir, self.competition)
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self
