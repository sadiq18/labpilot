"""Fetch competition overview + rules text (Plan 5b).

Primary path: Kaggle official ``competition_list_pages`` API (host-authored
Description / Evaluation / Rules / Code Requirements). Plain HTTP GETs of the
public SPA only return an empty JS shell, so they are a last-resort fallback.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("labpilot.competition.page_fetch")

# Below this length (after cleanup) we treat the page as a JS shell / empty.
_MIN_USEFUL_CHARS = 200

_SHELL_MARKERS = (
    "enable javascript",
    "you need to enable javascript",
    "noscript",
)

# Host page names → overview bundle (case-insensitive).
_OVERVIEW_PAGE_KEYS = frozenset(
    {
        "description",
        "evaluation",
        "code requirements",
        "code-requirements",
        "abstract",
        "data-description",
        "data description",
        "overview",
    }
)
_RULES_PAGE_KEYS = frozenset({"rules"})

ListPagesFn = Callable[[str], Sequence[Any]]


@dataclass(frozen=True)
class CompetitionPages:
    """Plain-text overview + rules for one competition slug."""

    slug: str
    overview_url: str
    rules_url: str
    overview_text: str
    rules_text: str
    is_empty_shell: bool
    source: str = "none"  # api | http | cache | none

    @property
    def combined_text(self) -> str:
        parts = [
            f"## Overview\n{self.overview_text}".strip(),
            f"## Rules\n{self.rules_text}".strip(),
        ]
        return "\n\n".join(p for p in parts if p and not p.endswith("\n"))


def competition_overview_url(slug: str) -> str:
    return f"https://www.kaggle.com/competitions/{slug}/overview"


def competition_rules_url(slug: str) -> str:
    return f"https://www.kaggle.com/competitions/{slug}/rules"


def html_to_text(html: str) -> str:
    """Strip scripts/styles/nav and return readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_content_to_text(content: str, *, mime_type: str = "") -> str:
    """Normalize API/HTML page bodies to plain text."""
    raw = (content or "").strip()
    if not raw:
        return ""
    mime = (mime_type or "").lower()
    looks_html = mime.startswith("text/html") or (
        raw.lstrip().startswith("<") and "</" in raw
    )
    if looks_html:
        return html_to_text(raw)
    return raw


def looks_like_empty_shell(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < _MIN_USEFUL_CHARS:
        return True
    lower = cleaned.lower()
    return any(marker in lower for marker in _SHELL_MARKERS) and len(cleaned) < 800


def fetch_page_text(url: str, *, timeout: float = 30.0) -> str:
    """GET a URL and return plain text, or '' on any failure."""
    if not url.strip():
        return ""
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        logger.info("Could not fetch competition page at %s.", url, exc_info=False)
        return ""
    return html_to_text(response.text)


def pages_from_api_payload(pages: Sequence[Any]) -> tuple[str, str]:
    """Split ``ApiCompetitionPage``-like objects into overview + rules text."""
    overview_parts: list[str] = []
    rules_parts: list[str] = []
    for page in pages:
        name = str(getattr(page, "name", "") or "").strip()
        if not name and isinstance(page, dict):
            name = str(page.get("name", "")).strip()
        content = getattr(page, "content", None)
        if content is None and isinstance(page, dict):
            content = page.get("content", "")
        mime = getattr(page, "mime_type", None) or ""
        if not mime and isinstance(page, dict):
            mime = page.get("mime_type", "") or page.get("mimeType", "")
        text = page_content_to_text(str(content or ""), mime_type=str(mime or ""))
        if not text:
            continue
        key = name.lower()
        heading = name or "Page"
        block = f"## {heading}\n{text}".strip()
        if key in _RULES_PAGE_KEYS:
            rules_parts.append(block)
        elif key in _OVERVIEW_PAGE_KEYS or key.replace(" ", "-") in {
            k.replace(" ", "-") for k in _OVERVIEW_PAGE_KEYS
        }:
            overview_parts.append(block)
        # Ignore prizes / timeline / other marketing pages for enrichment.
    return "\n\n".join(overview_parts).strip(), "\n\n".join(rules_parts).strip()


def fetch_competition_pages(
    slug: str,
    *,
    timeout: float = 30.0,
    knowledge_dir: Path | None = None,
    refresh: bool = False,
    list_pages: ListPagesFn | None = None,
) -> CompetitionPages:
    """Fetch overview + rules; optionally cache under ``research/raw/competitions/``.

    Prefer the authenticated Kaggle pages API. When ``knowledge_dir`` is set and
    ``refresh=False``, returns the latest cached text if present. ``refresh=True``
    re-fetches and appends a new RawStore version.
    """
    overview_url = competition_overview_url(slug)
    rules_url = competition_rules_url(slug)

    overview_text = ""
    rules_text = ""
    used_cache = False
    source = "none"

    if knowledge_dir is not None and not refresh:
        # Lazy import: RawStore lives under intelligence, which must not import
        # competition.page_fetch at package import time (circular).
        from labpilot.research_engine.intelligence.knowledge.sources import RawStore

        store = RawStore(knowledge_dir, slug)
        overview_text, rules_text, used_cache = _read_cache(store)
        if used_cache:
            source = "cache"

    if not used_cache:
        overview_text, rules_text, source = _fetch_live(
            slug,
            timeout=timeout,
            list_pages=list_pages,
        )
        if knowledge_dir is not None and (overview_text or rules_text):
            from labpilot.research_engine.intelligence.knowledge.sources import RawStore

            store = RawStore(knowledge_dir, slug)
            _write_cache(store, overview_text, rules_text, refresh=refresh)

    combined = f"{overview_text}\n{rules_text}".strip()
    empty = looks_like_empty_shell(combined) if combined else True
    return CompetitionPages(
        slug=slug,
        overview_url=overview_url,
        rules_url=rules_url,
        overview_text=overview_text,
        rules_text=rules_text,
        is_empty_shell=empty,
        source=source if (overview_text or rules_text) else "none",
    )


def _fetch_live(
    slug: str,
    *,
    timeout: float,
    list_pages: ListPagesFn | None,
) -> tuple[str, str, str]:
    overview_text, rules_text = _try_api_pages(slug, list_pages=list_pages)
    if overview_text or rules_text:
        return overview_text, rules_text, "api"

    overview_text = fetch_page_text(competition_overview_url(slug), timeout=timeout)
    rules_text = fetch_page_text(competition_rules_url(slug), timeout=timeout)
    if overview_text or rules_text:
        return overview_text, rules_text, "http"
    return "", "", "none"


def _try_api_pages(
    slug: str, *, list_pages: ListPagesFn | None
) -> tuple[str, str]:
    fetcher = list_pages
    if fetcher is None:
        from labpilot.diagnostics import kaggle_credentials_present

        if not kaggle_credentials_present():
            logger.info(
                "Skipping Kaggle pages API for %s — credentials not configured.",
                slug,
            )
            return "", ""
        fetcher = _default_list_pages

    try:
        pages = fetcher(slug)
    except Exception:
        logger.info(
            "Kaggle competition_list_pages failed for %s.", slug, exc_info=False
        )
        return "", ""
    if not pages:
        return "", ""
    return pages_from_api_payload(pages)


def _default_list_pages(slug: str) -> Sequence[Any]:
    from labpilot.config import KaggleConfig, Settings
    from labpilot.kaggle.client import KaggleClient

    settings = Settings()
    config = KaggleConfig(
        api_token=settings.kaggle_api_token,
        username=settings.kaggle_username,
        key=settings.kaggle_key,
    )
    api = KaggleClient(config).authenticate()
    return api.competition_list_pages(slug) or []


def _read_cache(store: Any) -> tuple[str, str, bool]:
    overview = store.latest("competitions", "overview")
    rules = store.latest("competitions", "rules")
    if overview is None and rules is None:
        return "", "", False
    overview_text = overview.path.read_text(encoding="utf-8") if overview else ""
    rules_text = rules.path.read_text(encoding="utf-8") if rules else ""
    return overview_text, rules_text, True


def _write_cache(
    store: Any, overview_text: str, rules_text: str, *, refresh: bool
) -> None:
    if overview_text:
        store.write(
            "competitions",
            "overview",
            overview_text,
            refresh=refresh,
            ext=".txt",
        )
    if rules_text:
        store.write(
            "competitions",
            "rules",
            rules_text,
            refresh=refresh,
            ext=".txt",
        )
