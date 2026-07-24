# Plan 6 — Paper Research Engine

Back to [Milestone 3](README.md). Design: README §4 · §2.4 Paper Understanding.

**Status:** Implemented. **Depends on:** Plans 1, 3 (Plan 2 for persist). **Unlocks:** Plan 8.

---

## Goal

Ship `PaperAnalyzer` + `LiteratureProvider` chain (Semantic Scholar + arXiv Search →
OpenAlex → arXiv PDF → Hugging Face Papers) and `PaperAnalyzerAgent` extract-not-summarize
into `PaperKnowledge` / `ResearchArtifact` cards. Fetch ≠ extract; cache under
`research/raw/papers/`.

## Why this matters

Papers are the primary external technique source. Structured cards feed the Knowledge Hub
and retrieval — not TL;DRs or chunk RAG.

## In scope

- LiteratureProvider interface + chain responsibilities (not fallbacks)
- Search/collect deterministic; Agent/rule_engine extract only after cache
- `PaperAnalyzerAgent` + `skill.md` → `PaperKnowledge`
- Soft-fail per provider
- Incremental RawStore catalog + PDF blobs (raise limit later without re-download)

## Out of scope

- Full-PDF essay summarization as product output
- GraphRAG entity discovery
- Hub merge across many papers (Plan 8)
- Non-arXiv Hugging Face attach (see Backlog)

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/literature/` | Models, clients, cache, query, chained provider |
| `analyzers/papers.py` | PaperAnalyzer |
| `micro_agents/paper_analyzer/` | Agent + skill.md → PaperKnowledge |
| Tests | Fixture API JSON; Agent rule_engine path |

## Decisions (shipped)

| Role | Backend |
|------|---------|
| Search | Semantic Scholar Graph API + arXiv Search (`arxiv` package, `delay_seconds=3`); merge/dedupe by DOI/arXiv id; OpenAlex search only if both empty (`SEMANTIC_SCHOLAR_API_KEY` optional; S2 429 → retry) |
| Enrich | OpenAlex Works API (`OPENALEX_MAILTO` / `OPENALEX_API_KEY` optional) |
| PDF | arXiv export API + PDF download into RawStore |
| Code / Hub | Hugging Face Papers API (replaces shut-down Papers with Code) |
| Extract | Abstract + metadata → `PaperKnowledge`; PDFs cached for future |
| Top-N | ≈15 mixed **recent** (about 70%, ≤3 years old) + **foundational** (about 30%); within bucket: relevance × citation velocity × `1/(1+age)^α` |
| Query | competition title/tags/`competition.json`; LLM rewrite when available |

## Acceptance criteria

- `research analyze papers <slug>` returns paper artifacts with techniques/claims fields.
- Re-run extract without re-fetch when raw cache present (`--refresh` forces fetch).
- No “summarize this paper” product path.

## Test plan

- Unit: provider normalize → Paper model.
- Unit: Agent/rule_engine → PaperKnowledge schema.
- Unit: catalog skip re-download; only new ids enrich.
- Optional live API tests marked, skipped in default CI.

## Review notes

- LLM never calls literature APIs directly; Deterministic Engine fetches.
- PwC is dead — HF Papers fills the code/Hub hop when `arxiv_id` is present.

## Backlog

- **Non-arXiv HF attach:** DOI / title lookup when papers lack an arXiv id (Hub filters /
  search), so code/model links are not limited to arXiv-indexed work.
