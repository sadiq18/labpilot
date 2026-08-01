# Backlog — Kuzu graph backend

**Status:** Backlog (M6+). M4 keeps logical SQL graph + abstract `GraphPort`.

## Problem

SQL-backed graph edges work for single-competition Context Engine quality.
Multi-hop reasoning and cross-competition transfer benefit from a graph-native
store.

## Proposed later work

- Implement `GraphPort` with **Kuzu** backend
- Migrate technique ↔ claim ↔ experiment ↔ paper traversals
- Keep Conductor / Context Engine callers on the port (no storage leaks)

## Signals from M4 (`ContextBundle.graph_metrics`)

Collect during expand/rank; revisit Kuzu when several hold over realistic campaigns:

| Signal | Suggests SQL strain |
|--------|---------------------|
| `neighbor_latency_ms_avg` / max | Joins too slow for interactive context |
| `slow_queries` (default ≥ 50ms) | Rising share of neighbor lookups |
| `neighbor_empty_results` / calls | Graph edges thin or queries wrong for multi-hop |
| `hop_depth_requested_max` ≥ 3 with high latency | Multi-hop SQL painful |
| `errors` | Timeouts / query failures under load |

Also qualitative: cannot express needed traversals without exploding SQL.

## Migration path

```text
M4: SQL-backed GraphPort + graph_metrics
  → M6+: Kuzu backend when signals justify it
  → Research Knowledge Graph OS
```

## Out of scope here

Introducing Kuzu or rewriting graph tables in M4.
