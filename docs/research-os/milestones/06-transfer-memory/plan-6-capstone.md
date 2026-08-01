# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end experience-memory smoke across two competitions: extract → store →
retrieve into ContextBundle → seed/inspect CLI; backlog links; post-M6 handoff
notes.

## Acceptance

- [x] Integration tests green: records from competition A surface for similar query on B
- [x] `research memory inspect --similar-to B` shows A-derived experiences when similar
- [x] `research memory seed --from A` (target B) is explicit and auditable
- [x] ContextBundle for B conduct/retrieve includes experience refs without auto-seeding campaign
- [x] Experience artifacts can key off `git_commit` when present (M5 handoff)
- [x] Warm-start reads durable experience store (+ graph/artifact links) — not git history alone
- [x] Backlog entries linked from M6 README
- [x] No silent auto-transfer; no Conductor bypass from memory subscribers

## Post-M6 handoff checklist

- [x] Automatic transfer with confidence scoring tracked in
      [automatic-transfer-confidence](../../backlog/automatic-transfer-confidence.md)
- [x] Emergent pattern extraction tracked in
      [experience-pattern-extraction](../../backlog/experience-pattern-extraction.md)
- [x] Shared/multi-tenant experience tables remain
      [shared-multi-tenant-store](../../backlog/shared-multi-tenant-store.md)
- [x] Hybrid ANN / Kuzu only when M4 metric signals justify
      ([hybrid-semantic-retrieval](../../backlog/hybrid-semantic-retrieval.md),
      [kuzu-graph-backend](../../backlog/kuzu-graph-backend.md))
- [x] Public memory-hierarchy ports still backlog
      ([memory-hierarchy-ports](../../backlog/memory-hierarchy-ports.md))
