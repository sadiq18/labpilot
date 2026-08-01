# Plan 6 — Capstone

Back to [README.md](README.md).

## Goal

End-to-end specialist runtime smoke: registry → Impl/Experiment → events → thin
parallel → git hash on artifacts; M6 handoff checklist; backlog links from M5 README.

## Acceptance

- [ ] Integration tests green (registry routing, experiment event, ≥2 parallel tasks, git hash)
- [ ] M6 handoff checklist written
- [ ] Backlog entries linked from M5 README
- [ ] Submit still gated; no ungated live Kaggle

## M6 handoff checklist

- [ ] Experience / transfer memory can key off experiment artifacts including `git_commit`
- [ ] Event bus subscribers must not bypass Conductor strategy when proposing next work
- [ ] Cross-competition warm-start reads durable artifacts + graph — not git history alone
- [ ] Full parallel research branches remain backlog until Campaign Engine v2 needs them
- [ ] When long-running multi-campaign orchestration starts, pull
      [async-conductor](../../backlog/async-conductor.md)
