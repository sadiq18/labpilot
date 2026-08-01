# M6 — Self-improving memory

Back to [../../README.md](../../README.md) · [execution-plan](../../execution-plan.md).

**Status:** Stub.  
**Branch:** `research-os-m6-transfer-memory`  
**Depends on:** M5 preferred  
**Design:** [10-memory-os](../../design/10-memory-os.md)

## Mission

Cross-competition experience: prompts, models, papers, augmentations, HPs;
warm-start new workspaces.

## Usable outcome

Second competition starts above blank slate.

## Tech that ships with M6

| Area | Technology |
|------|------------|
| Graph / vectors | Reuse M4 (Kuzu / Qdrant if present) |
| Analytics | DuckDB optional |

## Non-goals

- AutoML over all history
- Memory = vectors only
