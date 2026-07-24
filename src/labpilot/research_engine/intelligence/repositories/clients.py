"""Official GitHub REST API client for targeted repository collection."""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx

from labpilot.research_engine.intelligence.repositories.models import (
    RepoCategory,
    Repository,
)

_TIMEOUT = 60.0
_MIN_INTERVAL_S = 6.0
_MAX_ATTEMPTS = 7
_BACKOFF_BASE_S = 2.0
_BACKOFF_CAP_S = 60.0
_MAX_FILE_BYTES = 40_000


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(
        self,
        token: str = "",
        *,
        timeout: float = _TIMEOUT,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self.token = token.strip()
        self.timeout = timeout
        self.min_interval_s = max(0.0, min_interval_s)
        self._last_request_at = 0.0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "LabPilot/1.0 (research)",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.BASE}{path}"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._pace()
            try:
                response = httpx.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=self.timeout,
                    follow_redirects=True,
                )
                self._last_request_at = time.monotonic()
                retryable = response.status_code in {429, 502, 503, 504}
                retryable = retryable or (
                    response.status_code == 403
                    and (
                        response.headers.get("X-RateLimit-Remaining") == "0"
                        or bool(response.headers.get("Retry-After"))
                    )
                )
                if retryable and attempt < _MAX_ATTEMPTS:
                    time.sleep(_retry_delay(response, attempt))
                    continue
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                self._last_request_at = time.monotonic()
                if attempt >= _MAX_ATTEMPTS:
                    raise
                time.sleep(_backoff(attempt))
        raise RuntimeError("GitHub request exhausted retries")

    def search_repositories(
        self,
        query: str,
        *,
        category: RepoCategory,
        limit: int = 10,
    ) -> list[Repository]:
        data = self._get(
            "/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": max(1, min(limit, 100)),
            },
        )
        items = data.get("items") or []
        return [
            repo
            for idx, item in enumerate(items)
            if isinstance(item, dict)
            and (
                repo := self._normalize_search(
                    item,
                    category=category,
                    rank_hint=1.0 - idx / max(len(items), limit),
                )
            )
            is not None
        ]

    def get_repo(self, full_name: str) -> dict[str, Any]:
        return self._get(f"/repos/{full_name}")

    def get_readme(self, full_name: str) -> str:
        value = self._get(f"/repos/{full_name}/readme")
        return _decode_content(value)

    def get_tree(self, full_name: str, ref: str) -> list[str]:
        value = self._get(
            f"/repos/{full_name}/git/trees/{quote(ref, safe='')}",
            params={"recursive": "1"},
        )
        paths = [
            str(item["path"])
            for item in value.get("tree") or []
            if isinstance(item, dict) and item.get("type") == "blob" and item.get("path")
        ]
        return paths[:200]

    def get_file(self, full_name: str, path: str, *, ref: str) -> str:
        value = self._get(
            f"/repos/{full_name}/contents/{quote(path, safe='/')}",
            params={"ref": ref},
        )
        return _decode_content(value)[:_MAX_FILE_BYTES]

    def _normalize_search(
        self,
        item: dict[str, Any],
        *,
        category: RepoCategory,
        rank_hint: float,
    ) -> Repository | None:
        full_name = str(item.get("full_name") or "").strip()
        if not full_name:
            return None
        updated_at = _parse_datetime(item.get("updated_at"))
        topics = [str(v) for v in item.get("topics") or [] if str(v)]
        return Repository(
            id=f"github:{full_name.lower()}",
            full_name=full_name,
            url=str(item.get("html_url") or f"https://github.com/{full_name}"),
            description=str(item.get("description") or ""),
            stars=item.get("stargazers_count"),
            topics=topics,
            categories=[category],
            primary_category=category,
            language=item.get("language"),
            relevance=max(0.05, min(1.0, float(rank_hint))),
            updated_at=updated_at,
            payload={
                "github": {
                    "default_branch": item.get("default_branch") or "main",
                    "fork": bool(item.get("fork")),
                }
            },
        )


def _decode_content(value: dict[str, Any]) -> str:
    encoded = value.get("content")
    if not encoded:
        return ""
    try:
        return base64.b64decode(str(encoded), validate=False).decode(
            value.get("encoding") if value.get("encoding") not in {None, "base64"} else "utf-8",
            errors="replace",
        )
    except (ValueError, UnicodeError):
        return ""


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    return max(_MIN_INTERVAL_S, min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * 2 ** (attempt - 1)))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(_MIN_INTERVAL_S, float(retry_after))
        except ValueError:
            pass
    reset = response.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            return max(
                _MIN_INTERVAL_S,
                min(_BACKOFF_CAP_S, float(reset) - time.time()),
            )
        except ValueError:
            pass
    return _backoff(attempt)
