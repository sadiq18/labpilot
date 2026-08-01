"""Retrieve pipeline helpers — filter then BM25-score candidates."""

from __future__ import annotations

from labpilot.research_engine.context.bm25 import bm25_scores, tokenize
from labpilot.research_engine.context.filters import apply_filters
from labpilot.research_engine.context.models import ContextItem, ContextRequest
from labpilot.research_engine.context.retrieve_metrics import (
    LOW_TOP_SCORE,
    Bm25RetrieveMetrics,
)


def retrieve_candidates(
    items: list[ContextItem],
    request: ContextRequest,
) -> tuple[list[ContextItem], Bm25RetrieveMetrics]:
    """Filter by metadata, score with BM25, sort descending, apply max_items.

    Returns scored items plus metrics for BM25-vs-vector decisions.
    """
    metrics = Bm25RetrieveMetrics(candidates_in=len(items))
    filtered = apply_filters(items, request)
    metrics.candidates_after_filter = len(filtered)
    query = (request.query or request.goal or "").strip()
    q_tokens = tokenize(query)
    metrics.query_token_count = len(q_tokens)
    metrics.query_empty = not bool(q_tokens)

    if not filtered:
        metrics.candidates_out = 0
        metrics.no_positive_match = bool(q_tokens)
        metrics.notes.append("no candidates after filter")
        return [], metrics

    if query:
        metrics.bm25_applied = True
        texts = [f"{item.kind} {item.text} {item.reason}" for item in filtered]
        scores = bm25_scores(texts, query)
        raw_positive = sum(1 for s in scores if s > 0)
        metrics.scores_positive = raw_positive
        metrics.scores_zero = len(scores) - raw_positive
        metrics.no_positive_match = raw_positive == 0

        scored: list[ContextItem] = []
        for item, score in zip(filtered, scores, strict=True):
            combined = float(score) if score > 0 else float(item.score) * 0.01
            scored.append(
                item.model_copy(
                    update={
                        "score": combined,
                        "reason": (
                            f"{item.reason} | bm25={score:.4f}".strip(" |")
                            if item.reason
                            else f"bm25={score:.4f}"
                        ),
                    }
                )
            )
        scored.sort(key=lambda i: i.score, reverse=True)
        filtered = scored
    else:
        filtered = sorted(filtered, key=lambda i: i.score, reverse=True)
        metrics.notes.append("query empty; sorted by provider score")

    if request.max_items >= 0:
        filtered = filtered[: request.max_items]

    metrics.candidates_out = len(filtered)
    if filtered:
        metrics.top_score = float(filtered[0].score)
        if len(filtered) > 1:
            metrics.second_score = float(filtered[1].score)
        metrics.score_gap = metrics.top_score - metrics.second_score
        metrics.mean_kept_score = sum(i.score for i in filtered) / len(filtered)
        metrics.low_top_score = bool(query) and metrics.top_score < LOW_TOP_SCORE

        if q_tokens:
            top_text = " ".join(
                f"{i.kind} {i.text} {i.reason}" for i in filtered
            ).lower()
            hit = sum(1 for t in set(q_tokens) if t in top_text)
            metrics.query_terms_hit_in_topk = hit
            metrics.query_term_coverage = hit / len(set(q_tokens))

    return filtered, metrics
