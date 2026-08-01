# Backlog — Hybrid semantic retrieval

**Status:** Backlog (post-M4 / M6 scale). M4 ships BM25 + filters (+ later rank).

## Problem

Lexical BM25 is a strong deterministic baseline for research artifacts, but
semantic similarity (embeddings + ANN) and learned ranking improve recall when
wording diverges from stored text (paraphrase, synonym, cross-lingual notes).

## Proposed later work

- Embeddings via LLM router / provider
- **Qdrant** (or equivalent) when ANN is justified
- Hybrid BM25 + dense + SQL graph traversal
- Learned ranking over campaign outcomes

## Signals from M4 (`ContextBundle.bm25_metrics`)

Collect across real Conductor/CLI builds; revisit vectors when several hold
**and** operators confirm missed relevant evidence:

| Signal | Suggests BM25 strain |
|--------|----------------------|
| High rate of `no_positive_match` | Query terms never appear in corpus text |
| High rate of `low_top_score` | Best hit is weak even when non-zero |
| Low `query_term_coverage` on kept top-k | Query words absent from selected snippets |
| Tiny `score_gap` + wrong top item (human) | Ambiguous lexical ranking |
| Good BM25 scores but wrong semantics (human) | Need dense / hybrid — scores alone insufficient |

Do **not** migrate on a single threshold alone; pair metrics with labeled misses
or status/explain reviews.

## Migration path

```text
M4: BM25 + bm25_metrics
  → Hybrid BM25 + embeddings when signals + reviews justify
  → Qdrant / ANN at Research OS scale
```

## Out of scope here

Embeddings, Qdrant, hybrid ANN, or graph-neural retrieval in M4.
