"""Stage 1 — Intent Understanding (rules first; optional classify-only LLM)."""

from __future__ import annotations

import re
from typing import Any

from labpilot.accessor.common.micro_agents import StructuredContext
from labpilot.research_engine.intelligence.retrieval.models import QueryType, RetrievalIntent

_QUERY_TYPE_KEYWORDS: list[tuple[QueryType, tuple[str, ...]]] = [
    (QueryType.COMPARE, ("compare", "vs", "versus", "difference between")),
    (QueryType.EXPLAIN, ("explain", "why", "what does", "how does")),
    (
        QueryType.STRUCTURED_QUERY,
        ("find techniques", "which techniques", "list techniques", "techniques that"),
    ),
    (
        QueryType.HYPOTHESIS_GENERATION,
        ("improve", "suggest", "next experiment", "hypothesis", "how can i"),
    ),
]

_METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("macro_f1", re.compile(r"macro[\s_-]?f1", re.I)),
    ("cmap", re.compile(r"\bc-?map\b|\bcmap\b", re.I)),
    ("auc", re.compile(r"\bauc\b|roc[\s_-]?auc", re.I)),
    ("accuracy", re.compile(r"\baccuracy\b|\bacc\b", re.I)),
]

_DOMAIN_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("bioacoustics", ("bird", "audio", "soundscape", "bioacoustic", "clef")),
    ("computer_vision", ("image", "vision", "segmentation", "detection", "cell")),
    ("nlp", ("text", "nlp", "language", "llm")),
    ("tabular", ("tabular", "xgboost", "lightgbm")),
]


def classify_intent_rules(
    *,
    question: str = "",
    profile: dict[str, Any] | None = None,
    pipeline: list[str] | None = None,
    query_type: QueryType | str | None = None,
) -> RetrievalIntent:
    """Deterministic Intent classifier — zero latency, no LLM."""
    profile = profile or {}
    pipeline = [str(item).strip() for item in (pipeline or []) if str(item).strip()]
    text = (question or "").strip()
    lower = text.lower()

    resolved_type = _coerce_query_type(query_type) or _infer_query_type(lower)

    task = _first_str(profile.get("task"), profile.get("problem_type"))
    dataset = _first_str(profile.get("dataset"), profile.get("title"), profile.get("slug"))
    domain = _first_str(profile.get("domain")) or _infer_domain(lower, dataset or "")
    metric = _metric_from_profile(profile) or _infer_metric(lower)

    goal = None
    if "improve" in lower or resolved_type is QueryType.HYPOTHESIS_GENERATION:
        goal = f"Improve {metric}" if metric else "Improve competition score"
    elif resolved_type is QueryType.STRUCTURED_QUERY:
        goal = "Find relevant techniques"
    elif resolved_type is QueryType.EXPLAIN:
        goal = "Explain technique or result"
    elif resolved_type is QueryType.COMPARE:
        goal = "Compare techniques or pipelines"

    return RetrievalIntent(
        task=task or None,
        dataset=dataset or None,
        domain=domain or None,
        goal=goal,
        metric=metric or None,
        query_type=resolved_type,
        need_experiments=True,
        need_papers=True,
        need_repositories=resolved_type
        in {QueryType.HYPOTHESIS_GENERATION, QueryType.COMPARE},
        need_forums=False,
        current_pipeline=pipeline,
        question=text,
        classified_by="rules",
    )


def classify_intent(
    *,
    question: str = "",
    profile: dict[str, Any] | None = None,
    pipeline: list[str] | None = None,
    query_type: QueryType | str | None = None,
    llm_client: object | None = None,
) -> RetrievalIntent:
    """Rules-first classifier; optional LLM only for free-text gaps."""
    rules = classify_intent_rules(
        question=question,
        profile=profile,
        pipeline=pipeline,
        query_type=query_type,
    )
    # Strong structured path: explicit query_type and/or no free text → keep rules.
    if query_type is not None or not (question or "").strip() or llm_client is None:
        return rules

    from labpilot.research_engine.intelligence.micro_agents.intent_classifier import (
        IntentClassifierAgent,
    )

    agent = IntentClassifierAgent(llm_client=llm_client)
    try:
        result = agent.run(
            StructuredContext(
                text=question,
                data={
                    "profile": profile or {},
                    "pipeline": pipeline or [],
                    "query_type": str(query_type) if query_type else None,
                },
            )
        )
    except Exception:  # classify-only upgrade; rules always win on failure
        return rules

    if not isinstance(result, RetrievalIntent):
        return rules
    if not getattr(agent, "last_used_llm", False):
        return rules

    # Merge: LLM may fill gaps; explicit flags and pipeline from caller win.
    merged = rules.model_copy(deep=True)
    for field in ("task", "dataset", "domain", "goal", "metric"):
        llm_val = getattr(result, field, None)
        if llm_val and not getattr(merged, field):
            setattr(merged, field, llm_val)
    if result.query_type and query_type is None:
        merged.query_type = result.query_type
    if result.current_pipeline and not merged.current_pipeline:
        merged.current_pipeline = list(result.current_pipeline)
    merged.need_experiments = result.need_experiments
    merged.need_papers = result.need_papers
    merged.need_repositories = result.need_repositories
    merged.classified_by = "mixed"
    return merged


def _coerce_query_type(value: QueryType | str | None) -> QueryType | None:
    if value is None:
        return None
    if isinstance(value, QueryType):
        return value
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return QueryType(normalized)
    except ValueError:
        aliases = {
            "hypothesis": QueryType.HYPOTHESIS_GENERATION,
            "generate": QueryType.HYPOTHESIS_GENERATION,
            "structured": QueryType.STRUCTURED_QUERY,
            "query": QueryType.STRUCTURED_QUERY,
        }
        return aliases.get(normalized)


def _infer_query_type(lower: str) -> QueryType:
    for query_type, keywords in _QUERY_TYPE_KEYWORDS:
        if any(keyword in lower for keyword in keywords):
            return query_type
    return QueryType.HYPOTHESIS_GENERATION


def _infer_metric(lower: str) -> str | None:
    for name, pattern in _METRIC_PATTERNS:
        if pattern.search(lower):
            return name
    return None


def _infer_domain(lower: str, dataset: str) -> str | None:
    haystack = f"{lower} {dataset.lower()}"
    for domain, hints in _DOMAIN_HINTS:
        if any(hint in haystack for hint in hints):
            return domain
    return None


def _metric_from_profile(profile: dict[str, Any]) -> str | None:
    for key in ("metric", "evaluation_metric"):
        value = profile.get(key)
        if isinstance(value, dict):
            name = str(value.get("name") or "").strip()
            if name:
                return name
            continue
        text = _first_str(value)
        if text:
            return text
    return None


def _first_str(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text:
                    return text
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
