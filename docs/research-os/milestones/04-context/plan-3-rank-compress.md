# Plan 3 — Rank + compress

Back to [README.md](README.md).

## Goal

Real **rank** step (relevance, recency, cheap SQL graph distance) and
**compress** into a durable `ContextBundle` with token/budget caps.

## Acceptance

- [x] Rank reorders BM25 candidates with documented signals
- [x] Compress produces `ContextBundle` under budget
- [x] Bundle serializable for observe / CLI
- [x] Rank/expand calls `GraphPort.neighbors` and populate `ContextBundle.graph_metrics`
- [x] Document observed graph latency / empty / slow rates for Kuzu decision
  (see [kuzu backlog](../../backlog/kuzu-graph-backend.md))
