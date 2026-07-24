"""Incremental repository catalog cache on top of RawStore."""

from __future__ import annotations

import json
import re
from pathlib import Path

from labpilot.research_engine.intelligence.knowledge.sources import RawStore
from labpilot.research_engine.intelligence.repositories.models import Repository

_KIND = "repositories"
_CATALOG_INDEX = "catalog_index"


class RepoCatalogStore:
    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.store = RawStore(knowledge_dir, competition)

    def load_index(self) -> dict[str, str]:
        latest = self.store.latest(_KIND, _CATALOG_INDEX)
        if latest is None:
            return {}
        try:
            value = json.loads(latest.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}

    def _write_index(self, index: dict[str, str]) -> None:
        self.store.write(
            _KIND,
            _CATALOG_INDEX,
            json.dumps(index, indent=2, sort_keys=True) + "\n",
            refresh=True,
            ext=".json",
        )

    def load_repo(self, repo_id: str) -> Repository | None:
        latest = self.store.latest(_KIND, f"catalog__{_safe(repo_id)}")
        if latest is None:
            return None
        try:
            return Repository.model_validate_json(latest.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_repo(self, repo: Repository, *, refresh: bool = False) -> Repository:
        name = f"catalog__{_safe(repo.id)}"
        self.store.write(
            _KIND,
            name,
            repo.model_dump_json(indent=2) + "\n",
            refresh=refresh,
            ext=".json",
        )
        index = self.load_index()
        if repo.id not in index or refresh:
            index[repo.id] = name
            self._write_index(index)
        return repo

    def list_repos(self) -> list[Repository]:
        return [
            repo
            for repo_id in self.load_index()
            if (repo := self.load_repo(repo_id)) is not None
        ]

    def save_text(
        self,
        repo_id: str,
        path: str,
        text: str,
        *,
        refresh: bool = False,
    ) -> Path:
        name = f"text__{_safe(repo_id)}__{_safe(path)}"
        return self.store.write(
            _KIND, name, text, refresh=refresh, ext=".txt"
        ).path


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:180] or "repo"
