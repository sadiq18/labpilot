# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end specialist runtime smoke: registry → Impl/Experiment → events → thin
parallel → git hash on artifacts; M6 handoff checklist; backlog links from M5 README.

## Acceptance

- [x] Integration tests green (registry routing, experiment event, ≥2 parallel tasks, git hash)
- [x] M6 handoff checklist written
- [x] Backlog entries linked from M5 README
- [x] Submit still gated; no ungated live Kaggle

## M6 handoff checklist

Owned by M6 plans ([../06-transfer-memory/](../06-transfer-memory/)):

- [x] Experience / transfer memory can key off experiment artifacts including `git_commit`
      → M6 [plan-1](../06-transfer-memory/plan-1-experience-schema.md) /
      [plan-2](../06-transfer-memory/plan-2-experience-extractor.md) /
      [plan-6](../06-transfer-memory/plan-6-capstone.md)
- [x] Event bus subscribers must not bypass Conductor strategy when proposing next work
      → M6 [plan-5](../06-transfer-memory/plan-5-write-hooks.md) (write-only subscribers)
- [x] Cross-competition warm-start reads durable artifacts + graph — not git history alone
      → M6 ExperienceStore + [plan-3](../06-transfer-memory/plan-3-context-provider.md) /
      [plan-4](../06-transfer-memory/plan-4-memory-cli.md)

Still M5 / later:

- [x] Full parallel research branches remain backlog until Campaign Engine v2 needs them
- [x] When long-running multi-campaign orchestration starts, pull
      [async-conductor](../../backlog/async-conductor.md)

## Capstone test

[`tests/unit/test_agents_m5_capstone.py`](../../../../tests/unit/test_agents_m5_capstone.py) —
offline smoke over registry, Implementation + Experiment, Blinker evidence refresh,
thin parallel workers, and `git_commit` on `experiment/record.json`.
