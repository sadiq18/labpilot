"""Persistent per-provider rate/quota ledger.

Free tiers publish several independent limits (requests per minute, requests
per day, tokens per minute) and a campaign spans many processes, so the ledger
is on disk rather than in memory: a fresh `research conduct` must not forget
that it already spent today's allowance.

Pacing is proactive. Reacting to 429s spends quota on rejected calls and, on
most providers, lengthens the cooldown.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    provider   TEXT NOT NULL,
    ts         REAL NOT NULL,
    tokens     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_provider_ts ON llm_calls (provider, ts);
CREATE TABLE IF NOT EXISTS llm_cooldowns (
    provider   TEXT PRIMARY KEY,
    until_ts   REAL NOT NULL,
    reason     TEXT NOT NULL DEFAULT ''
);
"""

_MINUTE = 60.0
_DAY = 86_400.0


@dataclass(frozen=True)
class Availability:
    """Whether a provider can be called now, and if not, how long until it can."""

    ok: bool
    wait_seconds: float = 0.0
    reason: str = ""


class BudgetLedger:
    """Tracks spend per provider against published limits."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> BudgetLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def record(self, provider: str, *, tokens: int = 0, now: float | None = None) -> None:
        """Log a completed call. Call this even on failure — it consumed quota."""
        self._conn.execute(
            "INSERT INTO llm_calls (provider, ts, tokens) VALUES (?, ?, ?)",
            (provider, now if now is not None else time.time(), int(tokens)),
        )
        self._conn.commit()

    def cool_down(self, provider: str, seconds: float, reason: str = "429") -> None:
        """Mark a provider unavailable, honouring a server's Retry-After."""
        until = time.time() + max(0.0, seconds)
        self._conn.execute(
            "INSERT INTO llm_cooldowns (provider, until_ts, reason) VALUES (?, ?, ?) "
            "ON CONFLICT(provider) DO UPDATE SET until_ts=excluded.until_ts, "
            "reason=excluded.reason",
            (provider, until, reason),
        )
        self._conn.commit()

    def _count(self, provider: str, window: float, now: float) -> tuple[int, int]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(tokens), 0) AS t "
            "FROM llm_calls WHERE provider = ? AND ts > ?",
            (provider, now - window),
        ).fetchone()
        return int(row["n"]), int(row["t"])

    def _oldest_in_window(self, provider: str, window: float, now: float) -> float | None:
        row = self._conn.execute(
            "SELECT MIN(ts) AS oldest FROM llm_calls WHERE provider = ? AND ts > ?",
            (provider, now - window),
        ).fetchone()
        return float(row["oldest"]) if row and row["oldest"] is not None else None

    def availability(
        self,
        provider: str,
        *,
        rpm: int | None = None,
        rpd: int | None = None,
        tpm: int | None = None,
        now: float | None = None,
    ) -> Availability:
        """Can ``provider`` be called right now, and if not, when?"""
        now = now if now is not None else time.time()

        row = self._conn.execute(
            "SELECT until_ts, reason FROM llm_cooldowns WHERE provider = ?", (provider,)
        ).fetchone()
        if row and float(row["until_ts"]) > now:
            return Availability(
                False, float(row["until_ts"]) - now, f"cooling down ({row['reason']})"
            )

        if rpd is not None:
            used, _ = self._count(provider, _DAY, now)
            if used >= rpd:
                oldest = self._oldest_in_window(provider, _DAY, now)
                wait = (oldest + _DAY) - now if oldest else _DAY
                return Availability(False, max(wait, 0.0), f"daily limit {rpd} reached")

        if rpm is not None:
            used, _ = self._count(provider, _MINUTE, now)
            if used >= rpm:
                oldest = self._oldest_in_window(provider, _MINUTE, now)
                wait = (oldest + _MINUTE) - now if oldest else _MINUTE
                return Availability(False, max(wait, 0.0), f"rate limit {rpm}/min reached")

        if tpm is not None:
            _, tokens = self._count(provider, _MINUTE, now)
            if tokens >= tpm:
                oldest = self._oldest_in_window(provider, _MINUTE, now)
                wait = (oldest + _MINUTE) - now if oldest else _MINUTE
                return Availability(False, max(wait, 0.0), f"token limit {tpm}/min reached")

        return Availability(True)
