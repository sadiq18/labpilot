"""Dataset path / extension facet signals."""

from __future__ import annotations

import logging
from pathlib import Path

from labpilot.research_engine.memory.facets.context import FacetContext
from labpilot.research_engine.memory.facets.hints import CONF_DATASET, DATASET_HINTS
from labpilot.research_engine.memory.models import ExperienceFacet

logger = logging.getLogger(__name__)

_MAX_NAMES = 200


class DatasetExtractor:
    name = "dataset"

    def extract(self, ctx: FacetContext) -> list[ExperienceFacet]:
        names = _collect_names(ctx)
        if not names:
            return []
        corpus = " ".join(names).lower()
        hits: list[ExperienceFacet] = []
        for label, needles in DATASET_HINTS:
            matched = [n for n in needles if n in corpus]
            if not matched:
                continue
            hits.append(
                ExperienceFacet(
                    facet=label,
                    confidence=CONF_DATASET,
                    evidence=list(matched),
                    source="dataset",
                )
            )
        return hits


def _collect_names(ctx: FacetContext) -> list[str]:
    names: list[str] = []
    for key in ("data_paths", "dataset_paths", "files", "paths"):
        raw = ctx.payload.get(key)
        if isinstance(raw, (list, tuple)):
            names.extend(str(x) for x in raw)
        elif isinstance(raw, str) and raw.strip():
            names.append(raw)
    desc = ctx.payload.get("description")
    if isinstance(desc, str):
        names.append(desc)

    root = ctx.workspace_path
    if root is not None:
        try:
            for sub in ("data", "input", "datasets"):
                base = Path(root) / sub
                if not base.is_dir():
                    continue
                for path in base.rglob("*"):
                    if path.is_file():
                        names.append(str(path.relative_to(base)))
                    if len(names) >= _MAX_NAMES:
                        return names[:_MAX_NAMES]
        except OSError:
            logger.debug("dataset facet scan failed", exc_info=True)
    return names[:_MAX_NAMES]
