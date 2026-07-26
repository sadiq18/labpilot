# Plan 2 — EvidenceExtractor

Back to [Research Reflection](README.md). Design: [architecture.md](architecture.md).

**Status:** Ready. **Depends on:** Plan 1. **Unlocks:** Plans 3–5.

---

## Goal

Deterministic extraction of structured evidence from an Engineer execution
(metrics, config, runtime, comparison) — **no LLM**.

## In scope

- `reflection/evidence/extractor.py`
- Inputs: execution workspace evidence pack, optional `experiments/comparator`
  output, plan/hypothesis ids
- Persist via `ReflectionStore` → `experiment_evidence`
- Strength heuristic (rules): metric delta vs baseline / failure → strong|moderate|weak|rejected

## Out of scope

- Critic narrative
- Belief mutation

## Acceptance criteria

- [ ] Given fixture execution artifacts, produces stable evidence JSON
- [ ] Offline / no network / no LLM
- [ ] Unit tests with fixture competition metrics
