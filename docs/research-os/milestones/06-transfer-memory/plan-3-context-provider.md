# Plan 3 — Context Engine experience provider

Back to [README.md](README.md).

**Status:** Done.

## Goal

Add an **experience provider** to the M4 Context Engine pipeline so retrieve →
rank → compress can include similar cross-competition Experience Records in
`ContextBundle`.

```text
build_context(request)
  → … existing providers …
  → experience provider (filter + BM25 / tag overlap / outcome)
  → rank → compress → ContextBundle
```

Conductor continues to consume `ContextBundle` only. Experience never schedules
tasks or overrides approvals. Memory **influences**; it does not control.

## Acceptance

- [x] Experience provider registered in Context Engine retrieve path
- [x] Similar experiences appear in ContextBundle for a new competition / query
- [x] Ranking respects budgets (`max_items` / `max_chars`); experiences compressible
- [x] No path from provider to Task Queue / Conductor schedule (observe-only via bundle)
- [x] Unit/integration tests: store has records from slug A; retrieve for slug B surfaces them when similar
- [x] Explain/retrieve paths can cite experience refs (reuse M4 explain patterns where applicable)

## Implementation notes

- `ExperienceProvider` in `context/providers/experience.py`; registered in `default_providers`
- Metadata uses `source_competition` (not `competition`) so filters keep cross-comp items
- Operator seeds boost score via `memory/seeds/*.json`

## Out of scope

- `research memory` CLI (plan 4)
- Write hooks (plan 5)
- Automatic campaign-start seeding
- Embedding/ANN hybrid ([backlog](../../backlog/hybrid-semantic-retrieval.md))
- Confidence-scored auto-transfer ([backlog](../../backlog/automatic-transfer-confidence.md))
