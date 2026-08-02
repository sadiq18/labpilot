"""Bounded inputs for artifact-aware facet extractors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FacetContext:
    """Read-only bundle passed to every :class:`FacetExtractor`."""

    competition: str
    payload: dict[str, Any] = field(default_factory=dict)
    hypothesis_text: str = ""
    action: str = ""
    reflection: dict[str, Any] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    workspace_path: Path | None = None
    paper_texts: list[str] = field(default_factory=list)
