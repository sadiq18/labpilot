# M4 — Memory & Context Engine

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Implementing (plans 1–2 done; next plan 3).  
**Branch:** `research-os-m4-context`  
**Depends on:** M3  
**Design:** [07-context-engine](../../design/07-context-engine.md) · [10-memory-os](../../design/10-memory-os.md)

## Mission

Intelligence layer: Context Engine (`retrieve → rank → compress → ContextBundle`).
Wire into Conductor; CLI for trust/debug. Keep Evidence Card / Research Graph as
semantic seed. Memory hierarchy stays **internal** (public ports = backlog).

## Usable outcome

Task-local `ContextBundle`s for Conductor/CLI; foundation for `explain`.

## Phase plans

| # | Plan | Focus |
|---|------|--------|
| 1 | [plan-1-context-skeleton.md](plan-1-context-skeleton.md) | `context/` package, ports, AnyIO facade, RI provider |
| 2 | [plan-2-retrieve-bm25.md](plan-2-retrieve-bm25.md) | Multi-source retrieve + BM25 + filters |
| 3 | [plan-3-rank-compress.md](plan-3-rank-compress.md) | Real rank + compress → ContextBundle |
| 4 | [plan-4-conductor-wire.md](plan-4-conductor-wire.md) | Conductor observe/policy consumes ContextBundle |
| 5 | [plan-5-cli-explain.md](plan-5-cli-explain.md) | retrieve/explain CLI for trust/debug |
| 6 | [plan-6-capstone.md](plan-6-capstone.md) | Integration tests + M5 handoff |

**Order:** plan-1 → … → plan-6.

## Tech that ships with M4

| Area | Technology |
|------|------------|
| Package | `research_engine/context/` orchestration |
| Metadata | SQLite (existing knowledge + conductor DBs) |
| Retrieval | Filters + **BM25**; **`bm25_metrics`** on bundles to judge vector/hybrid later |
| Rank / compress | Real rank + budgeted compress (plan 3) |
| Graph | Logical SQL + `GraphPort`; **`graph_metrics`** on bundles to judge Kuzu later |
| Vectors | **Defer** (backlog) |
| Runtime | **AnyIO** inside context retrieve; Conductor stays **sync** |

## Non-goals

- Public memory-hierarchy API ([backlog](../../backlog/memory-hierarchy-ports.md))
- Embeddings / Qdrant / hybrid ANN ([backlog](../../backlog/hybrid-semantic-retrieval.md))
- Kuzu migration ([backlog](../../backlog/kuzu-graph-backend.md))
- Agent registry / parallel fan-out (M5)
- Rewriting Plan 9 RI retrieval
- Making Conductor async
