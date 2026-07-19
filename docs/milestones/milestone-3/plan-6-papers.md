# Plan 6 — Paper Research Engine

Back to [Milestone 3](README.md). Design: README §4 · §2.4 Paper Understanding.

**Status:** Not started. **Depends on:** Plans 1, 3 (Plan 2 for persist). **Unlocks:** Plan 8.

---

## Goal

Ship `PaperAnalyzer` + `LiteratureProvider` chain (Semantic Scholar → OpenAlex → arXiv →
Papers with Code) and `PaperAnalyzerAgent` extract-not-summarize into `PaperKnowledge` /
`ResearchArtifact` cards. Fetch ≠ extract; cache under `research/raw/papers/`.

## Why this matters

Papers are the primary external technique source. Structured cards feed the Knowledge Hub
and retrieval — not TL;DRs or chunk RAG.

## In scope

- LiteratureProvider interface + chain responsibilities (not fallbacks)
- Search/collect deterministic; Agent/rule_engine extract only after cache
- `PaperAnalyzerAgent` + `skill.md` (contributions, methods, limitations, ideas)
- Soft-fail per provider

## Out of scope

- Full-PDF essay summarization as product output
- GraphRAG entity discovery
- Hub merge across many papers (Plan 8)

## Implementation checklist

| Path | Work |
|------|------|
| `analyzers/papers.py`, `analyzers/literature/*` | Provider + analyzer |
| `micro_agents/paper_analyzer/` | Agent + skill.md |
| Tests | Fixture API JSON; Agent rule_engine path |

## Acceptance criteria

- `research analyze papers <slug>` returns paper artifacts with techniques/claims fields.
- Re-run extract without re-fetch when raw cache present (`--refresh` forces fetch).
- No “summarize this paper” product path.

## Test plan

- Unit: provider normalize → Paper model.
- Unit: Agent/rule_engine → PaperKnowledge schema.
- Optional live API tests marked, skipped in default CI.

## Review notes

- LLM never calls literature APIs directly; Deterministic Engine fetches.
