# Backlog — Experience pattern extraction

**Status:** Backlog (post-M6). M6 stores structured Experience Records only;
retrieval decides what to surface. Do not ship first-class prompt/HP/paper/feature
wiki tables in M6.

## Problem

Over time, repeated successful (and failed) experiences imply reusable patterns:

```text
Experience Memory → extract → Prompt | Model | Feature | Paper patterns
```

Hardcoding those categories as M6 stores is rigid. Patterns should emerge from
retrieval and usage statistics once enough experiences exist.

## Proposed later work

- Aggregate experiences by tags / artifact similarity / outcome
- Optional DuckDB (or equivalent) offline analysis over ExperienceStore
- Materialize pattern summaries as derived views — not a replacement SoR
- Feed pattern summaries back into Context Engine as an additional provider
- Keep Experience Records as the durable episode log

## Out of scope here

M6 Experience Record schema, extractor, context provider, CLI
([06-transfer-memory](../milestones/06-transfer-memory/) ·
[10-memory-os](../design/10-memory-os.md)).
