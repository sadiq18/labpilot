# Plan 2 — Knowledge Store (SQLite + research/ tree)

Back to [Milestone 3](README.md). Design: [knowledge-system.md](knowledge-system.md) · README §8, §11.

**Status:** Not started. **Depends on:** Plan 1. **Unlocks:** Plans 8–11.

---

## Goal

Persist explored intelligence under `knowledge/<slug>/research/` with distinct
`raw/` → `extracted/` → `knowledge/` layers and a per-competition `knowledge.db` (SQLite
ontology + join tables). Storage API only — no LLM, no retrieval ranking.

## Why this matters

Scripts become a platform only when knowledge is queryable and versionable. This plan is the
SoR everything else writes into and Plan 9 queries.

## In scope

- Directory layout: `raw/`, `extracted/`, `knowledge/`, `experiments/`, `reports/`,
  `embeddings/` (empty stub), `knowledge.db`
- Immutable / versioned-append rules for `raw/`
- Tables: `research_artifacts`, `techniques`, `datasets`, `architectures`, `tasks`,
  `references` / join tables (`paper_techniques`, `experiment_techniques`, …), `hypotheses`,
  `beliefs`, `findings`, `experiments`
- Store API: upsert artifact, merge technique stub, list by type
- DB wins for joins; `reports/analyze.json` remains projection (writers may still be stub)

## Out of scope

- GraphRAG / Neo4j (Appendix A — Future)
- Embeddings population (Future Stage 3)
- Knowledge Extraction hub merge logic (Plan 8)
- Multi-stage retrieval (Plan 9)

## Design summary

- Ontology-first graph stored relationally (Appendix A).
- Re-extract rebuilds `extracted/` from `raw/` without re-fetch when blobs exist.

## Implementation checklist

| Path | Work |
|------|------|
| `intelligence/knowledge/store.py` | Paths + SQLite |
| `intelligence/knowledge/schema.sql` or migrations | Schema |
| `intelligence/cache.py` or sources helpers | raw/ write-once |
| Tests | Temp dir DB fixtures |

## Acceptance criteria

- Creating a competition research root yields the locked tree.
- Upsert `ResearchArtifact` → row in `research_artifacts` + JSON under `extracted/`.
- Join: technique ↔ papers/experiments via relationship tables returns expected ids.
- `--refresh` adds a new raw version rather than silent overwrite (documented API).
- No Neo4j/vector dependency.

## Test plan

- Unit: schema migrate, upsert, query SpecAugment-style join.
- Unit: raw immutability / version append.
- No network.

## Review notes

- Confirm paths match knowledge-system.md (not legacy `intelligence/` on-disk tree).
- JSONL-as-SoR must not reappear.
