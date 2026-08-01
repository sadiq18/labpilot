# Plan 2 — Retrieve (BM25 + filters)

Back to [README.md](README.md).

## Goal

Multi-source retrieve into ranked candidate bags:

- Metadata filters (competition, kind, status)
- **BM25** over candidate text (deterministic, no embeddings)
- Sources: RI symbolic hits, workspace notes/artifacts, experiment/evidence
  summaries, Conductor decision/feedback snippets

## Acceptance

- [ ] BM25 scores candidates from multiple providers
- [ ] Filters exclude out-of-scope items
- [ ] Unit tests with fixture documents (offline)
