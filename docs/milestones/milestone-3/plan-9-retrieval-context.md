# Plan 9 — Retrieval + Context Builder

Back to [Milestone 3](README.md). Design: README §9 · knowledge-system.md §5–5f.

**Status:** Not started. **Depends on:** Plan 8. **Unlocks:** Plan 10.

---

## Goal

Ship multi-stage retrieval (Intent → Symbolic → Evidence Expansion → Context Compression)
and `ContextBuilder` producing typed `ResearchContext` (L1–L3 budgets). LLM never sees the
database. Fixed `QueryPlan` per `query_type`; no embeddings Stage 3 in this plan.

## Why this matters

This is how knowledge reaches reasoning without RAG noise. Progressive Context and
compression are the difference between a research platform and chat-over-chunks.

## In scope

- `RetrievalIntent` classifier (rules first; optional small LLM classify-only)
- Symbolic SQL/facet retrieval + pipeline-diff (similar pipelines → missing techniques)
- Evidence expansion along `references`
- Context compression to technique cards
- `ContextBuilder.build` → `ResearchContext` → serializable brief
- Hierarchical memory budgets (L1–L3); L4 never dumped
- Fixed QueryPlan stubs for Hypothesis Generation / structured query

## Out of scope

- Semantic ranking / embeddings (Future)
- Rich adaptive Query Planner (Future)
- Hypothesis Assistant product ranking (Plan 10)
- GraphRAG

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/retrieve.py` | Multi-stage |
| `intelligence/context_builder.py` (or under retrieve/) | Builder |
| Models: RetrievalIntent, ResearchContext, QueryPlan | |
| Tests | Join query Macro F1/Audio; compression size bound |

## Acceptance criteria

- Structured query “techniques improve Macro F1 on Audio with ≥N papers/experiments”
  answered via SQL joins without embeddings.
- `ContextBuilder` output is typed `ResearchContext`; unit tests assert field presence.
- Compressed brief excludes raw PDF/thread text.
- No LLM required for symbolic+compress path.

## Test plan

- Unit: intent from competition profile rules.
- Unit: symbolic filters + expansion.
- Unit: token/char budget on compressed cards.
- No network.

## Review notes

- LLM is last / optional consumer — not implemented as product until Plan 10.
- Confirm forbidden: embed corpus first; dump L4.
