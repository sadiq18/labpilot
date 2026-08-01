# Plan 7 — Research Claims

Back to [Research Reflection](README.md). Design: [beliefs-and-claims.md](beliefs-and-claims.md).

**Status:** Done. **Depends on:** Plans 1, 4, 6. **Unlocks:** Plan 8 journal claims section.

---

## Goal

First-class `research_claims` (+ `claim_evidence`) with promotion from strong
beliefs; contradictions tracked.

## In scope

- `claims/promoter.py` + models
- Promotion rules from [beliefs-and-claims.md](beliefs-and-claims.md)
- Optional CLI `research claims list|show` (or fold into journal first)
- Use `evidence_links` / claim edges so belief-graph UI can project later

## Out of scope

- Full belief-graph UI (follow-on / Plan 7b)
- Multi-agent debate

## Acceptance criteria

- [x] Claims queryable distinct from beliefs
- [x] Promotion creates claim_evidence edges
- [x] Contested path when contradicting evidence appears
