"""Build literature search queries from competition context (Plan 6)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from labpilot.llm.client import LLMClient
from labpilot.research_engine.intelligence.models import AnalyzeContext

logger = logging.getLogger("labpilot.research_engine.intelligence.literature.query")


def build_literature_query(
    context: AnalyzeContext,
    *,
    llm_client: LLMClient | None = None,
    competitions_dir: Path | None = None,
) -> list[str]:
    """Return 1–3 search strings (deterministic seed; optional LLM rewrite)."""
    seed = _deterministic_seed(context, competitions_dir=competitions_dir)
    if llm_client is None:
        return seed
    try:
        rewritten = _llm_rewrite(llm_client, context.competition, seed)
        if rewritten:
            return rewritten
    except Exception:
        logger.info("Literature query LLM rewrite failed; using deterministic seed.")
    return seed


def _deterministic_seed(
    context: AnalyzeContext, *, competitions_dir: Path | None
) -> list[str]:
    meta = _load_competition_meta(context, competitions_dir=competitions_dir)
    title = str(meta.get("title") or "").strip()
    tags = [str(t).strip() for t in (meta.get("tags") or []) if str(t).strip()]
    modality = str(meta.get("modality") or meta.get("problem_type") or "").strip()
    parts = [p for p in [title, *tags[:5], modality] if p]
    if not parts:
        # Slug → rough keywords: birdclef-2026 → birdclef 2026
        slug_words = re.sub(r"[-_]+", " ", context.competition).strip()
        parts = [slug_words] if slug_words else [context.competition]
    primary = " ".join(dict.fromkeys(parts))
    queries = [primary]
    if title and tags:
        queries.append(f"{title} {' '.join(tags[:3])}")
    if modality and title:
        queries.append(f"{modality} {title}")
    # Deduplicate while preserving order.
    return list(dict.fromkeys(q.strip() for q in queries if q.strip()))[:3]


def _load_competition_meta(
    context: AnalyzeContext, *, competitions_dir: Path | None
) -> dict[str, Any]:
    # Prefer latest run competition.json
    runs = context.runs_dir
    if runs.is_dir():
        candidates = sorted(
            runs.glob("*/competition.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Prefer same competition slug in path when present.
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            slug = str(data.get("slug") or data.get("competition") or "")
            if slug and slug != context.competition:
                continue
            return _meta_from_mapping(data)
        for path in candidates[:3]:
            try:
                return _meta_from_mapping(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue

    # YAML under configs/competitions
    root = competitions_dir or Path("configs/competitions")
    for name in (f"{context.competition}.yaml", f"{context.competition}.yml"):
        path = root / name
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                return _meta_from_mapping(data)
        except Exception:
            continue
    return {"title": context.competition.replace("-", " ")}


def _meta_from_mapping(data: dict[str, Any]) -> dict[str, Any]:
    tags = data.get("tags") or data.get("keywords") or []
    if isinstance(tags, str):
        tags = [tags]
    return {
        "title": data.get("title") or data.get("name") or "",
        "tags": list(tags),
        "modality": data.get("modality") or "",
        "problem_type": data.get("problem_type") or "",
    }


def _llm_rewrite(client: LLMClient, competition: str, seed: list[str]) -> list[str]:
    system = (
        "You rewrite competition context into 1-3 short academic literature "
        "search queries for Semantic Scholar. Respond ONLY with a JSON array "
        'of strings, e.g. ["query one", "query two"]. No markdown.'
    )
    user = (
        f"Competition slug: {competition}\n"
        f"Seed queries: {json.dumps(seed)}\n"
        "Produce focused ML/research queries (techniques, modalities, benchmarks)."
    )
    raw = client.complete(system, user)
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if isinstance(parsed, list):
        out = [str(x).strip() for x in parsed if str(x).strip()]
        return out[:3] if out else []
    return []
