"""Bounded, deterministic retrieval of context for planning.

Loads only what the compiler needs to draft a plan — the hypothesis, a few
beliefs, a few relevant techniques, and a short Research Brief excerpt — under
fixed budgets. No LLM here. Domain reads (KnowledgeStore / brief file) are
imported lazily so this module has no hard coupling to the intelligence pillar's
internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labpilot.research_engine.shared.experiments.models import Hypothesis
from labpilot.research_engine.intelligence.paths import ResearchPaths

#: Retrieval budgets (README L1–L3): keep context small and stable.
MAX_BELIEFS = 8
MAX_TECHNIQUES = 8
MAX_BRIEF_CHARS = 1500


@dataclass
class RetrievedContext:
    hypothesis: Hypothesis
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    techniques: list[dict[str, Any]] = field(default_factory=list)
    brief_excerpt: str = ""


def retrieve(
    hypothesis: Hypothesis,
    *,
    knowledge_dir: Path,
    competition: str,
    knowledge_store: Any | None = None,
) -> RetrievedContext:
    """Assemble a bounded context bundle for one hypothesis."""
    beliefs: list[dict[str, Any]] = []
    techniques: list[dict[str, Any]] = []

    store = knowledge_store
    owns_store = False
    if store is None:
        from labpilot.research_engine.intelligence.knowledge.store import KnowledgeStore

        store = KnowledgeStore(knowledge_dir, competition)
        owns_store = True
    try:
        try:
            beliefs = list(store.list_beliefs())[:MAX_BELIEFS]
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            beliefs = []
        try:
            techniques = list(store.list_techniques(limit=MAX_TECHNIQUES))
        except Exception:  # noqa: BLE001 - retrieval is best-effort
            techniques = []
    finally:
        if owns_store:
            store.close()

    return RetrievedContext(
        hypothesis=hypothesis,
        beliefs=beliefs,
        techniques=techniques,
        brief_excerpt=_brief_excerpt(knowledge_dir, competition),
    )


def _brief_excerpt(knowledge_dir: Path, competition: str) -> str:
    paths = ResearchPaths(knowledge_dir, competition)
    brief_path = paths.brief_path
    if not brief_path.is_file():
        return ""
    text = brief_path.read_text(errors="ignore").strip()
    return text[:MAX_BRIEF_CHARS]
