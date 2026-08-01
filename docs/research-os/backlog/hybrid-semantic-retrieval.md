# Backlog — Hybrid semantic retrieval

**Status:** Backlog (post-M4 / M6 scale). M4 ships BM25 + filters + rank only.

## Problem

Lexical BM25 is a strong deterministic baseline for research artifacts, but
semantic similarity (embeddings + ANN) and learned ranking improve recall when
wording diverges from stored text.

## Proposed later work

- Embeddings via LLM router / provider
- **Qdrant** (or equivalent) when ANN is justified
- Hybrid BM25 + dense + SQL graph traversal
- Learned ranking over campaign outcomes

## Out of scope here

Embeddings, Qdrant, hybrid ANN, or graph-neural retrieval in M4.
