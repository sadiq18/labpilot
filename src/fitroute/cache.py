"""SQLite-backed prompt cache for LLM completions."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def cache_key(model: str, prompt: str, temperature: float, system: str) -> str:
    """Stable SHA256 key matching the plan: model+prompt+temperature+system."""
    material = f"{model}\n{temperature}\n{system}\n{prompt}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class PromptCache:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._conn: sqlite3.Connection | None = None
        # Same reasoning as `BudgetLedger`, which fixed this and left its
        # sibling behind: one gateway per process ("one per process is plenty"),
        # touched by whichever thread makes an LLM call — the proxy's worker
        # threads, or anything else that ends up off the thread that built it.
        #
        # The cost of missing it is total rather than partial: `_complete_once`
        # reads the cache *before* selecting a provider, so a cross-thread call
        # dies at the lookup and every micro agent on that thread falls back at
        # once. Measured on rogii 2026-08-08 — CodeEngineerAgent,
        # EvidenceSynthesisAgent and RecommendationAgent all "skipped" for this
        # one line, and codegen silently rendered a Jinja template instead of
        # calling the model.
        self._lock = threading.RLock()
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get(self, key: str) -> str | None:
        if not self.enabled or self._conn is None:
            return None
        # The lock lives here, not in the caller: `check_same_thread=False`
        # makes cross-thread use *possible*, and only serialising makes it
        # *safe*. sqlite tolerates cross-thread use, not concurrent use.
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        logger.debug("LLM cache hit: %s", key[:12])
        return str(row[0])

    def set(self, key: str, response: str, *, model: str) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO llm_cache (cache_key, response, model)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response = excluded.response,
                    model = excluded.model,
                    created_at = datetime('now')
                """,
                (key, response, model),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
