# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end experience-memory smoke across two competitions: extract → store →
retrieve into ContextBundle → seed/inspect CLI; backlog links; post-M6 handoff
notes.

## Acceptance

- [ ] Integration tests green: records from competition A surface for similar query on B
- [ ] `research memory inspect --similar-to B` shows A-derived experiences when similar
- [ ] `research memory seed --from A` (target B) is explicit and auditable
- [ ] ContextBundle for B conduct/retrieve includes experience refs without auto-seeding campaign
- [ ] Experience artifacts can key off `git_commit` when present (M5 handoff)
- [ ] Warm-start reads durable experience store (+ graph/artifact links) — not git history alone
- [ ] Backlog entries linked from M6 README
- [ ] No silent auto-transfer; no Conductor bypass from memory subscribers

## Post-M6 handoff checklist

- [ ] Automatic transfer with confidence scoring tracked in
      [automatic-transfer-confidence](../../backlog/automatic-transfer-confidence.md)
- [ ] Emergent pattern extraction tracked in
      [experience-pattern-extraction](../../backlog/experience-pattern-extraction.md)
- [ ] Shared/multi-tenant experience tables remain
      [shared-multi-tenant-store](../../backlog/shared-multi-tenant-store.md)
- [ ] Hybrid ANN / Kuzu only when M4 metric signals justify
      ([hybrid-semantic-retrieval](../../backlog/hybrid-semantic-retrieval.md),
      [kuzu-graph-backend](../../backlog/kuzu-graph-backend.md))
- [ ] Public memory-hierarchy ports still backlog
      ([memory-hierarchy-ports](../../backlog/memory-hierarchy-ports.md))
