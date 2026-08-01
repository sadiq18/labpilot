# Plan 3 — Rank + compress

Back to [README.md](README.md).

## Goal

Real **rank** step (relevance, recency, cheap SQL graph distance) and
**compress** into a durable `ContextBundle` with token/budget caps.

## Acceptance

- [ ] Rank reorders BM25 candidates with documented signals
- [ ] Compress produces `ContextBundle` under budget
- [ ] Bundle serializable for observe / CLI
- [ ] Rank/expand calls `GraphPort.neighbors` and populate `ContextBundle.graph_metrics`
- [ ] Document observed graph latency / empty / slow rates for Kuzu decision
  (see [kuzu backlog](../../backlog/kuzu-graph-backend.md))
