"""Build an :class:`AnalyzeContext` from CLI inputs (design §3.5 input row).

Accepts a competition slug *or* a Kaggle competition URL and normalizes both to
a slug — there is no second entrypoint for URLs, just this thin normalize step.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from labpilot.research_engine.intelligence.models import AnalyzeContext


def normalize_competition(value: str) -> tuple[str, str | None]:
    """Return ``(slug, url)`` for a slug or Kaggle competition URL.

    >>> normalize_competition("birdclef-2026")
    ('birdclef-2026', None)
    >>> normalize_competition("https://www.kaggle.com/competitions/birdclef-2026")
    ('birdclef-2026', 'https://www.kaggle.com/competitions/birdclef-2026')
    """
    raw = value.strip()
    if not raw:
        raise ValueError("Competition slug or URL must not be empty.")

    if "://" not in raw and "kaggle.com" not in raw:
        return raw.strip("/"), None

    url = raw if "://" in raw else f"https://{raw}"
    parts = [p for p in urlparse(url).path.split("/") if p]
    # .../competitions/<slug>[/...] or .../c/<slug>
    for marker in ("competitions", "c"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1], url
    if parts:
        return parts[-1], url
    raise ValueError(f"Could not parse a competition slug from {value!r}.")


def build_context(
    competition: str,
    *,
    runs_dir: Path,
    knowledge_dir: Path,
    refresh: bool = False,
    data_dir: Path | None = None,
) -> AnalyzeContext:
    slug, url = normalize_competition(competition)
    return AnalyzeContext(
        competition=slug,
        runs_dir=runs_dir,
        knowledge_dir=knowledge_dir,
        refresh=refresh,
        url=url,
        data_dir=data_dir,
    )
