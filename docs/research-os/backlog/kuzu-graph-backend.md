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

Populate during rank/expand when `build_context` calls `GraphPort.neighbors`
for top retrieve seeds. Inspect bundle notes / `LABPILOT_DEBUG_METRICS=1` output
after realistic campaigns; revisit Kuzu when several hold:

| Signal | Suggests SQL strain |
|--------|---------------------|
| `neighbor_latency_ms_avg` / max | Joins too slow for interactive context |
| `slow_queries` (default ≥ 50ms) | Rising share of neighbor lookups |
| `neighbor_empty_results` / calls | Graph edges thin or queries wrong for multi-hop |
| `hop_depth_requested_max` ≥ 3 with high latency | Multi-hop SQL painful |
| `errors` | Timeouts / query failures under load |

Also qualitative: cannot express needed traversals without exploding SQL.

### How to observe (operators)

```text
LABPILOT_DEBUG_METRICS=1  →  [context] … | graph neighbors=N returned=… empty=… slow=… latency_*
ContextBundle.graph_metrics  →  same counters on the durable bundle (JSON via to_json())
```

Empty / high-latency rates over many builds are the decision inputs — not a single run.

## Migration path

```text
M4: SQL-backed GraphPort + graph_metrics
  → M6+: Kuzu backend when signals justify it
  → Research Knowledge Graph OS
```

## Out of scope here

Introducing Kuzu or rewriting graph tables in M4.
