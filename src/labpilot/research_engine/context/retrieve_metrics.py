"""Metrics for BM25 retrieve — signals when lexical search is not enough."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Top BM25 score below this suggests weak lexical match (proxy only).
LOW_TOP_SCORE = 1.0


class Bm25RetrieveMetrics(BaseModel):
    """Counters from one filter+BM25 pass.

    Use these (across many builds) to decide whether to add embeddings / ANN
    (see hybrid-semantic-retrieval backlog). Proxies only — confirm with
    labeled or operator-judged misses.
    """

    candidates_in: int = 0
    candidates_after_filter: int = 0
    candidates_out: int = 0
    query_empty: bool = False
    bm25_applied: bool = False
    query_token_count: int = 0
    scores_positive: int = 0
    scores_zero: int = 0
    top_score: float = 0.0
    second_score: float = 0.0
    score_gap: float = 0.0
    mean_kept_score: float = 0.0
    query_terms_hit_in_topk: int = 0
    query_term_coverage: float = 0.0
    low_top_score: bool = False
    no_positive_match: bool = False
    notes: list[str] = Field(default_factory=list)
