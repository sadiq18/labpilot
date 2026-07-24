"""Incremental paper catalog + PDF cache on top of RawStore."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from labpilot.research_engine.intelligence.knowledge.sources import RawStore
from labpilot.research_engine.intelligence.literature.models import Paper

logger = logging.getLogger("labpilot.research_engine.intelligence.literature.cache")

_CATALOG_INDEX = "catalog_index"
_KIND = "papers"


class PaperCatalogStore:
    """Versioned paper catalog + PDF blobs under ``research/raw/papers/``.

    - Catalog entries are keyed by stable ``Paper.id`` (``doi:`` / ``arxiv:`` / ``s2:``).
    - Existing entries are reused unless ``refresh=True`` (append new RawStore version).
    - Raising search limits later only downloads **missing** ids.
    """

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.store = RawStore(knowledge_dir, competition)
        self.competition = competition
        self.knowledge_dir = knowledge_dir

    def load_index(self) -> dict[str, str]:
        """Map paper_id → catalog blob name (usually the safe paper id)."""
        latest = self.store.latest(_KIND, _CATALOG_INDEX)
        if latest is None:
            return {}
        try:
            data = json.loads(latest.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    def _write_index(self, index: dict[str, str], *, refresh: bool) -> None:
        payload = json.dumps(index, indent=2, sort_keys=True) + "\n"
        self.store.write(_KIND, _CATALOG_INDEX, payload, refresh=refresh, ext=".json")

    def has(self, paper_id: str) -> bool:
        return self.load_paper(paper_id) is not None

    def load_paper(self, paper_id: str) -> Paper | None:
        name = f"catalog__{_safe(paper_id)}"
        latest = self.store.latest(_KIND, name)
        if latest is None:
            return None
        try:
            return Paper.model_validate_json(latest.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_paper(self, paper: Paper, *, refresh: bool = False) -> Paper:
        name = f"catalog__{_safe(paper.id)}"
        payload = paper.model_dump_json(indent=2) + "\n"
        self.store.write(_KIND, name, payload, refresh=refresh, ext=".json")
        index = self.load_index()
        if paper.id not in index or refresh:
            index[paper.id] = name
            self._write_index(index, refresh=True)
        return paper

    def list_papers(self) -> list[Paper]:
        papers: list[Paper] = []
        for paper_id in self.load_index():
            paper = self.load_paper(paper_id)
            if paper is not None:
                papers.append(paper)
        return papers

    def pdf_name(self, paper_id: str) -> str:
        return f"pdf__{_safe(paper_id)}"

    def has_pdf(self, paper_id: str) -> bool:
        return self.store.latest(_KIND, self.pdf_name(paper_id)) is not None

    def load_pdf_path(self, paper_id: str) -> Path | None:
        latest = self.store.latest(_KIND, self.pdf_name(paper_id))
        return latest.path if latest else None

    def save_pdf(self, paper_id: str, data: bytes, *, refresh: bool = False) -> Path:
        version = self.store.write(
            _KIND, self.pdf_name(paper_id), data, refresh=refresh, ext=".pdf"
        )
        return version.path

    def merge_or_fetch(
        self,
        candidates: list[Paper],
        *,
        refresh: bool = False,
    ) -> list[Paper]:
        """Return papers preferring cache; mark which ids were already local.

        Does not perform HTTP — callers enrich only papers missing from cache
        (unless refresh).
        """
        merged: list[Paper] = []
        for candidate in candidates:
            cached = None if refresh else self.load_paper(candidate.id)
            if cached is not None:
                # Prefer fresher search relevance; keep enriched fields from cache.
                updated = cached.model_copy(deep=True)
                updated.relevance = max(updated.relevance, candidate.relevance)
                if candidate.abstract and len(candidate.abstract) > len(updated.abstract):
                    updated.abstract = candidate.abstract
                merged.append(updated)
            else:
                merged.append(candidate)
        return merged


def _safe(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")[:180]
        or "paper"
    )
