"""Bounded workspace code scan for import / API facet signals."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import CODE_HINTS, CONF_CODE, confidence_from_hits
from labpilot.research_engine.memory.models import ExperienceFacet

logger = logging.getLogger(__name__)

_MAX_FILES = 24
_MAX_BYTES = 40_000
_CODE_NAMES = {
    "train.py",
    "model.py",
    "models.py",
    "infer.py",
    "dataset.py",
    "augment.py",
    "augmentations.py",
    "requirements.txt",
    "pyproject.toml",
}


class CodeExtractor:
    name = "code"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        if ctx.workspace_path is None:
            return []
        try:
            text = _read_code_corpus(Path(ctx.workspace_path))
        except OSError:
            logger.debug("code facet scan failed", exc_info=True)
            return []
        if not text:
            return []
        corpus = text.lower()
        hits: list[ExperienceFacet] = []
        for label, needles in CODE_HINTS:
            matched = [n for n in needles if n in corpus]
            if not matched:
                continue
            conf = max(CONF_CODE, confidence_from_hits(len(matched)))
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=min(conf, 0.9),
                    evidence=list(matched),
                    source="code",
                )
            )
        return hits


def _read_code_corpus(root: Path) -> str:
    candidates: list[Path] = []
    for sub in ("pipeline", "src", "."):
        base = root / sub if sub != "." else root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.name.lower() in _CODE_NAMES or path.suffix.lower() in {
                ".py",
                ".toml",
                ".txt",
            }:
                if "site-packages" in path.parts or ".venv" in path.parts:
                    continue
                candidates.append(path)
            if len(candidates) >= _MAX_FILES:
                break
        if len(candidates) >= _MAX_FILES:
            break
    parts: list[str] = []
    for path in candidates[:_MAX_FILES]:
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace")[:_MAX_BYTES])
        except OSError:
            continue
    return "\n".join(parts)
