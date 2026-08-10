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


def store_is_absent(knowledge_dir, competition: str) -> bool:
    """Has *anything* been written for this competition yet?

    The question every "is there work to do" helper needs, and none of them
    asked. They wrapped a store read in `except Exception: return None` with a
    comment saying *"absent store means nothing yet"* — true of the case they
    had in mind, and false of every other one. A locked database, a schema the
    code no longer matches, a permissions problem: all became "no plans", "no
    hypotheses", "nothing measured", and the conductor acted on that. An error
    that is indistinguishable from an answer is the M20 shape one layer up from
    the gates.

    Asked before the read, so absence returns cleanly and the handler is left
    holding only genuine faults — which are then worth logging loudly, because
    they mean something.
    A workspace with no knowledge directory at all counts as absent, which is
    what it is: nothing has been written, because there is nowhere to write it.
    Said explicitly because `Path(None)` raises, and a `TypeError` escaping here
    would crash the conductor at a call site whose whole purpose is to answer a
    question calmly.
    """
    from pathlib import Path

    if not knowledge_dir:
        return True
    return not ResearchPaths(Path(knowledge_dir), competition).db_path.is_file()


def hypotheses_are_absent(knowledge_dir, competition: str) -> bool:
    """Has any hypothesis been written for this competition yet?

    Separate from `store_is_absent` because `HypothesisStore` is file-backed,
    not SQLite — asking about `knowledge.db` would answer *"no hypotheses"* for
    a workspace that has a directory full of them. The absence check has to name
    the store it is standing in for, or it becomes the same mistake in the other
    direction.
    """
    from pathlib import Path

    from labpilot.workspace import competition_data_root

    if not knowledge_dir:
        return True
    return not (competition_data_root(Path(knowledge_dir), competition) / "hypotheses").is_dir()
