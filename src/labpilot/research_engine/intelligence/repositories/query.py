"""Build category-aware GitHub search plans from competition context."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.llm.client import LLMClient
from labpilot.research_engine.intelligence.micro_agents.repo_query_planner import (
    RepoQueryPlannerAgent,
)
from labpilot.research_engine.intelligence.models import AnalyzeContext
from labpilot.research_engine.intelligence.repositories.models import (
    RepoCategory,
    RepoSearchPlan,
    RepoSearchQuery,
)

_STOP = {
    "during",
    "the",
    "a",
    "an",
    "of",
    "and",
    "for",
    "to",
    "in",
    "on",
    "with",
    "from",
    "by",
}


def build_repo_queries(
    context: AnalyzeContext,
    *,
    llm_client: LLMClient | None = None,
    competitions_dir: Path | None = None,
) -> RepoSearchPlan:
    meta = _load_meta(context, competitions_dir=competitions_dir)
    title = str(meta.get("title") or "").strip()
    tags = [str(v).strip() for v in meta.get("tags", []) if str(v).strip()]
    modality = str(meta.get("modality") or meta.get("problem_type") or "").strip()
    keywords = _keywords(context.competition, title=title, tags=tags)
    core = " ".join(keywords[:3]) or context.competition.replace("-", " ")
    seed = _seed_plan(context.competition, core=core, modality=modality)
    agent = RepoQueryPlannerAgent(llm_client=llm_client)
    result = agent.run(
        StructuredContext(
            competition=context.competition,
            text=json.dumps(
                {
                    "title": title,
                    "tags": tags,
                    "modality": modality,
                    "keywords": keywords,
                    "core": core,
                }
            ),
            data={"seed_queries": [q.model_dump(mode="json") for q in seed.queries]},
        )
    )
    if isinstance(result, RepoSearchPlan) and result.queries:
        cleaned = _sanitize_plan(result, fallback=seed)
        if cleaned.queries:
            return cleaned
    return seed


def broaden_repo_queries(competition: str) -> RepoSearchPlan:
    """Looser fallback queries when the primary plan returns zero hits."""
    keywords = _keywords(competition)
    core = " ".join(keywords[:3]) or competition.replace("-", " ")
    slug_words = competition.replace("-", " ")
    return RepoSearchPlan(
        queries=[
            RepoSearchQuery(
                category=RepoCategory.BASELINE,
                query=f"{core} language:Python",
            ),
            RepoSearchQuery(
                category=RepoCategory.WINNING_SOLUTION,
                query=f"{slug_words} solution",
            ),
            RepoSearchQuery(
                category=RepoCategory.DOMAIN_LIBRARY,
                query=f"{core} library language:Python",
            ),
            RepoSearchQuery(
                category=RepoCategory.OTHER,
                query=slug_words,
            ),
        ]
    )


def _seed_plan(competition: str, *, core: str, modality: str = "") -> RepoSearchPlan:
    slug_words = competition.replace("-", " ")
    modality_bit = f" {modality}" if modality else ""
    templates = (
        (RepoCategory.WINNING_SOLUTION, f"{slug_words} solution"),
        (RepoCategory.WINNING_SOLUTION, f"{core} kaggle solution language:Python"),
        (RepoCategory.BASELINE, f"{slug_words} baseline"),
        (RepoCategory.BASELINE, f"{core} starter language:Python"),
        (RepoCategory.DOMAIN_LIBRARY, f"{core}{modality_bit} library language:Python"),
        (RepoCategory.TRAINING_PIPELINE, f"{core} pytorch training language:Python"),
        (RepoCategory.AUGMENTATION, f"{core} augmentation language:Python"),
    )
    return RepoSearchPlan(
        queries=[
            RepoSearchQuery(category=category, query=query.strip())
            for category, query in templates
        ][:8]
    )


def _sanitize_plan(plan: RepoSearchPlan, *, fallback: RepoSearchPlan) -> RepoSearchPlan:
    """Drop over-constrained LLM queries (too many required phrases)."""
    kept: list[RepoSearchQuery] = []
    for item in plan.queries:
        query = item.query.strip()
        if not query:
            continue
        # Multiple quoted phrases AND many tokens usually return zero GitHub hits.
        quoted = len(re.findall(r'"[^"]+"', query))
        tokens = len(query.replace('"', "").split())
        if quoted >= 2 and tokens >= 6:
            continue
        if tokens > 10:
            continue
        kept.append(item)
    return RepoSearchPlan(queries=kept[:8]) if kept else fallback


def _keywords(competition: str, *, title: str = "", tags: list[str] | None = None) -> list[str]:
    parts: list[str] = []
    parts.extend(re.split(r"[-_\s]+", competition.lower()))
    if title:
        parts.extend(re.split(r"[-_\s]+", title.lower()))
    for tag in tags or []:
        parts.extend(re.split(r"[-_\s]+", tag.lower()))
    out: list[str] = []
    for part in parts:
        token = re.sub(r"[^a-z0-9]+", "", part)
        if not token or token in _STOP or token.isdigit():
            continue
        if token not in out:
            out.append(token)
    return out


def _load_meta(
    context: AnalyzeContext,
    *,
    competitions_dir: Path | None,
) -> dict[str, Any]:
    if context.runs_dir.is_dir():
        candidates = sorted(
            context.runs_dir.glob("*/competition.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            slug = str(data.get("slug") or data.get("competition") or "")
            if not slug or slug == context.competition:
                return _normalize_meta(data)
    root = competitions_dir or Path("configs/competitions")
    for suffix in ("yaml", "yml"):
        path = root / f"{context.competition}.{suffix}"
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return _normalize_meta(data)
        except (OSError, ValueError):
            continue
    return {"title": re.sub(r"[-_]+", " ", context.competition)}


def _normalize_meta(data: dict[str, Any]) -> dict[str, Any]:
    tags = data.get("tags") or data.get("keywords") or []
    if isinstance(tags, str):
        tags = [tags]
    return {
        "title": data.get("title") or data.get("name") or "",
        "tags": list(tags),
        "modality": data.get("modality") or "",
        "problem_type": data.get("problem_type") or "",
    }
