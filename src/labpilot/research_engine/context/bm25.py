"""In-repo BM25 (Okapi) for deterministic lexical scoring."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over an in-memory tokenized corpus."""

    def __init__(
        self,
        corpus: list[list[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                self.df[term] += 1
        self.n_docs = len(corpus)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: list[str], index: int) -> float:
        if not self.corpus or not query or index < 0 or index >= self.n_docs:
            return 0.0
        freqs = self.doc_freqs[index]
        dl = self.doc_len[index]
        score = 0.0
        avgdl = self.avgdl or 1.0
        for term in query:
            if term not in freqs:
                continue
            tf = freqs[term]
            denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            score += self.idf(term) * (tf * (self.k1 + 1.0)) / denom
        return float(score)

    def scores(self, query: list[str]) -> list[float]:
        return [self.score(query, i) for i in range(self.n_docs)]


def bm25_scores(texts: list[str], query: str) -> list[float]:
    """Score each text against ``query``; zeros if query empty."""
    q_tokens = tokenize(query)
    if not q_tokens or not texts:
        return [0.0] * len(texts)
    corpus = [tokenize(t) for t in texts]
    ranker = BM25(corpus)
    return ranker.scores(q_tokens)
