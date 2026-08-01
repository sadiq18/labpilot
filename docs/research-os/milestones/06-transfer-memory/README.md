# M6 — Self-improving memory

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Implementing (plans 1–2 done; next plan 3).  
**Branch:** `research-os-m6-transfer-memory`  
**Depends on:** M4 **hard** (Context Engine); M5 **preferred** (`git_commit` on
artifacts, event hooks for write path)  
**Design:** [10-memory-os](../../design/10-memory-os.md) · Context:
[07-context-engine](../../design/07-context-engine.md)

Handoff from M5: [../05-agents/plan-6-capstone.md](../05-agents/plan-6-capstone.md).

## Mission

Build **research memory** as structured **Experience Records** across competitions.
Retrieval decides what to surface into Conductor context. Human CLI for seed/inspect.
Do not ship a curated prompt/HP/paper wiki.

## Usable outcome

```text
New Competition
        |
        v
Context Engine ── retrieve similar experiences ──► ContextBundle
        |
        v
Conductor → Research Campaign

Optional:
  research memory seed --from <slug>
  research memory inspect --similar-to <slug>
```

Second competition can warm-start above blank slate via retrieve (+ optional seed).
Memory influences decisions; never silently controls them.

## Phase plans

| # | Plan | Focus | Status |
|---|------|--------|--------|
| 1 | [plan-1-experience-schema.md](plan-1-experience-schema.md) | Experience Record model + SQLite ExperienceStore | Done |
| 2 | [plan-2-experience-extractor.md](plan-2-experience-extractor.md) | Deterministic extract: experiment/reflection → record | Done |
| 3 | [plan-3-context-provider.md](plan-3-context-provider.md) | Context Engine experience provider → ContextBundle | |
| 4 | [plan-4-memory-cli.md](plan-4-memory-cli.md) | `research memory seed` / `inspect` (+ list/show) | |
| 5 | [plan-5-write-hooks.md](plan-5-write-hooks.md) | Persist on completion; idempotent upsert | |
| 6 | [plan-6-capstone.md](plan-6-capstone.md) | Integration smoke + backlog handoff | |

**Order:** plan-1 → … → plan-6.

## Experience Record (first cut)

```text
goal | hypothesis | action | result | outcome(success|fail)
artifacts: experiment, metrics, reflection, git_commit (when M5)
tags: modality / technique facets (not a taxonomy product)
```

Later (not M6 stores): experience → emergent prompt / model / feature / paper
patterns — see [experience-pattern-extraction](../../backlog/experience-pattern-extraction.md).

## Tech that ships with M6

| Area | Technology |
|------|------------|
| Package | Experience store + extractor under research engine (exact layout in plans) |
| Metadata | Shared SQLite `experiences.db` (not under competition `knowledge/`) |
| Path resolution | env → `labpilot.yaml` → parent research root → `~/.labpilot` |
| Retrieve | M4 Context Engine provider (filters + BM25; graph when useful) |
| Write hooks | M5 Blinker subscriber preferred; Reflection/Engineer fallback |
| CLI | `research memory` (seed / inspect / list / show) |
| Graph / vectors / DuckDB | Reuse M4; **no** new engines required for M6 exit |

```text
~/kaggle/
  experiences.db              ← transferable memory (shared)
  birdclef-2026/
    knowledge/research/knowledge.db   ← competition SoR (flat; no nested slug)
  titanic/
    knowledge/research/knowledge.db
```

## Backlog (deferred from M6)

| Item | Why deferred |
|------|----------------|
| [automatic-transfer-confidence.md](../../backlog/automatic-transfer-confidence.md) | Auto warm-start with confidence — M7+; avoid hidden bias |
| [experience-facet-extraction.md](../../backlog/experience-facet-extraction.md) | Facet stages 2–5 (Stage 1 confidence+evidence shipped) |
| [experience-pattern-extraction.md](../../backlog/experience-pattern-extraction.md) | Emergent pattern libraries from usage — after memory foundation |
| [shared-multi-tenant-store.md](../../backlog/shared-multi-tenant-store.md) | Scale shared tables beyond single-machine SQLite |
| [hybrid-semantic-retrieval.md](../../backlog/hybrid-semantic-retrieval.md) | Embeddings when BM25 gaps justify |
| [kuzu-graph-backend.md](../../backlog/kuzu-graph-backend.md) | Graph-native backend when SQL hurts |
| [memory-hierarchy-ports.md](../../backlog/memory-hierarchy-ports.md) | Public tier ports still not required |

Index: [../../backlog/README.md](../../backlog/README.md).

## Non-goals

- AutoML over all history
- Memory = vectors only
- First-class prompt / HP / architecture / paper / feature wiki tables
- DuckDB analytics as M6 gate
- Silent automatic seeding at campaign start
- Bypassing Conductor strategy from memory subscribers
- Replacing Evidence Card / Research Graph / Reflection SoR
