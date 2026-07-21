"""Raw source store — Layer 1 immutable originals (knowledge-system.md §4).

``raw/`` holds original blobs (PDFs, README dumps, discussion JSON). Blobs are
**immutable**: a normal write never overwrites an existing version. ``--refresh``
(``refresh=True``) appends a *new* version instead of silently replacing the old
one, so re-extraction can always rebuild ``extracted/`` from ``raw/`` and history
is never lost.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from labpilot.research_engine.intelligence.paths import RAW_SUBDIRS, ResearchPaths

_INDEX_NAME = "versions.json"
_BLOB_PREFIX = "v"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "source"


@dataclass(frozen=True)
class RawVersion:
    kind: str
    name: str
    version: int
    path: Path
    sha256: str
    created_at: str


class RawStore:
    """Versioned, write-once blob storage under ``research/raw/<kind>/<name>/``."""

    def __init__(self, knowledge_dir: Path, competition: str) -> None:
        self.paths = ResearchPaths(knowledge_dir, competition)
        self.paths.raw_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, kind: str, name: str) -> Path:
        if kind not in RAW_SUBDIRS:
            raise ValueError(f"kind must be one of {RAW_SUBDIRS}, got {kind!r}")
        return self.paths.raw_dir / kind / _safe_name(name)

    def _index_path(self, kind: str, name: str) -> Path:
        return self._entry_dir(kind, name) / _INDEX_NAME

    def _load_index(self, kind: str, name: str) -> list[dict]:
        index_path = self._index_path(kind, name)
        if not index_path.is_file():
            return []
        return json.loads(index_path.read_text())

    def versions(self, kind: str, name: str) -> list[RawVersion]:
        entry_dir = self._entry_dir(kind, name)
        return [
            RawVersion(
                kind=kind,
                name=name,
                version=item["version"],
                path=entry_dir / item["file"],
                sha256=item["sha256"],
                created_at=item["created_at"],
            )
            for item in self._load_index(kind, name)
        ]

    def latest(self, kind: str, name: str) -> RawVersion | None:
        versions = self.versions(kind, name)
        return versions[-1] if versions else None

    def write(
        self,
        kind: str,
        name: str,
        data: bytes | str,
        *,
        refresh: bool = False,
        ext: str = "",
    ) -> RawVersion:
        """Store a blob and return its version.

        - First write for ``(kind, name)`` → version 1.
        - Existing versions and ``refresh=False`` → **no overwrite**; returns the
          latest existing version unchanged (idempotent).
        - ``refresh=True`` → appends a new version.
        """
        existing = self.versions(kind, name)
        if existing and not refresh:
            return existing[-1]

        entry_dir = self._entry_dir(kind, name)
        entry_dir.mkdir(parents=True, exist_ok=True)

        payload = data.encode() if isinstance(data, str) else data
        version = (existing[-1].version + 1) if existing else 1
        suffix = ext if ext.startswith(".") or not ext else f".{ext}"
        file_name = f"{_BLOB_PREFIX}{version:04d}{suffix}"
        (entry_dir / file_name).write_bytes(payload)

        record = {
            "version": version,
            "file": file_name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "created_at": datetime.now(UTC).isoformat(),
        }
        index = self._load_index(kind, name)
        index.append(record)
        self._index_path(kind, name).write_text(json.dumps(index, indent=2) + "\n")

        return RawVersion(
            kind=kind,
            name=name,
            version=version,
            path=entry_dir / file_name,
            sha256=record["sha256"],
            created_at=record["created_at"],
        )

    def read(self, kind: str, name: str, *, version: int | None = None) -> bytes | None:
        versions = self.versions(kind, name)
        if not versions:
            return None
        if version is None:
            target = versions[-1]
        else:
            matches = [v for v in versions if v.version == version]
            if not matches:
                return None
            target = matches[0]
        return target.path.read_bytes()
