# Plan 9 — Submission + Reporting / Memory

Back to [Research Engineer](README.md). Design: [capabilities.md](capabilities.md) ·
[architecture.md](architecture.md) §8.

**Status:** Done. **Depends on:** Plan 8. **Unlocks:** Plans 10–11.

---

## Goal

Implement **Submission** (package + optional Kaggle upload) and **Reporting / Memory**
(experiment report, Knowledge/plan reflection writes). Upload is capability-owned and
deterministic; never LLM-triggered.

## In scope

- `capabilities/submission/` — wrap `research_engine/submission/` / Kaggle client usage
- `capabilities/reporting/` — markdown/JSON report under execution or experiments dir;
  Knowledge Store / plan reflection updates as designed
- Flags: dry-run upload vs real upload (safe default dry-run in tests)
- Evidence: submission path, Kaggle submission id, report path

## Out of scope

- Deleting Pipeline (Plan 10)
- Full unattended capstone (Plan 11)

## Acceptance criteria

- Submit task packages prediction file; upload gated by explicit config/flag
- Report task writes durable report + memory hooks without LLM inventing metrics
- Soft-fail upload errors are typed (retry vs fail)

## Test plan

- Unit: package without upload
- Unit: report writes expected files
- Integration: eval → submit(dry) → report mini DAG
