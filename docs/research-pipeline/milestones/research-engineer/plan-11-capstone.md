# Plan 11 — Capstone: unattended baseline experiment

Back to [Research Engineer](README.md). Design: [README.md](README.md) §3–4 ·
[baseline-plan.md](baseline-plan.md).

**Status:** Done. **Depends on:** Plans 1–10. **Unlocks:** Milestone 5 complete.

---

## Goal

Prove the success criterion: after Analyze, create **P-001** baseline, leave the machine,
and get a verified experiment (smoke → train → eval → submit package → optional Kaggle
upload → report/reflect) without babysitting the queue.

## Why this matters

This is the product bar for Autonomous Research Engineer — not a unit-test-only milestone.

## In scope

- End-to-end on a chosen competition (document which; prefer one already used in Analyze
  fixtures such as biohub-cell-tracking-during-development or a smaller tabular fixture)
- Script or documented command sequence:

  ```bash
  research analyze <competition>
  research plan create <competition> --baseline
  research run --plan P-001
  # optional: research resume …
  ```

- Capstone notes under `docs/research-pipeline/milestones/research-engineer/` (results paths, known gaps)
- Update `IN-PROGRESS.md` / `MILESTONES.md` status when green

## Out of scope

- Hypothesis-driven improvement loop beyond baseline (future milestone)
- Perfect LB score

## Acceptance criteria

- Smoke gate runs before train
- Artifacts + `experiments` row + execution `E-xxx` terminal status
- Report written; plan/tasks reflect completion
- Upload: dry-run acceptable in CI; real upload documented if credentials available
- Operator can walk away after `run` starts (no required mid-flight prompts for happy path)

## Test plan

- Manual capstone checklist in the notes doc
- Automate what is safe (dry-run, local Runtime) in CI if feasible
