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

- [x] Experience / transfer memory can key off experiment artifacts including `git_commit`
- [x] Event bus subscribers must not bypass Conductor strategy when proposing next work
- [x] Cross-competition warm-start reads durable artifacts + graph — not git history alone
- [x] Full parallel research branches remain backlog until Campaign Engine v2 needs them
- [x] When long-running multi-campaign orchestration starts, pull
      [async-conductor](../../backlog/async-conductor.md)

## Capstone test

[`tests/unit/test_agents_m5_capstone.py`](../../../../tests/unit/test_agents_m5_capstone.py) —
offline smoke over registry, Implementation + Experiment, Blinker evidence refresh,
thin parallel workers, and `git_commit` on `experiment/record.json`.
