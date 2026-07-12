# TODO — Planned Milestones

Back to [MILESTONES.md](../MILESTONES.md).

Items here are explicitly **not** in the current implementation scope. Build only after P1 ships.

---

## P2 — v0.3 (Remote Runtime & Scheduling) — **Deferred**

> Superseded by P3 priority. Remote runtime work resumes after the iteration loop ships.

**Goal:** Offload `train_model` to user-registered remote runtimes with quota-aware scheduling.

- Register Kaggle notebooks/kernels, Google Colab, and other providers in `configs/runtimes/`
- `RuntimeProfiler` tracks free/paid usage limits locally
- `RuntimeScheduler` picks a runtime within quotas (`--remote-train`, `--runtime <id>`)
- Poll remote job status until artifacts sync back; `research resume` re-enters poll loop
- Init and post-train stages stay local in v1

```bash
research build --run-id <id> --remote-train
research runtime list | register | doctor
```

---

## P3 — v0.4 (Iteration Loop) — **Done**

See [COMPLETED.md](COMPLETED.md).

---

## P4 — v1.0 (Production Quality)

**Goal:** Reliable tool for repeated competition use.

- Multi-competition project workspace
- Config overrides (`--config`, `--dry-run`, `--submit`)
- CI-tested templates per problem type

---

## Future (Explicitly Deferred)

| Capability | Why deferred |
|------------|--------------|
| Multi-agent systems | Orchestrator + templates are enough for P0/P1 |
| Vector databases | Brief uses competition page + profiler, not retrieval |
| Knowledge graphs | No cross-competition reasoning needed yet |
| Long-term memory | Each run is self-contained |
| Autonomous planning | Fixed pipeline DAG is sufficient |
| Self-modifying code | Templates + parameterization first |
| AutoML search | One strong baseline proves the loop |
| Multi-model orchestration | Single model per run |
| Full-pipeline remote execution | Training-only remote in P2 v1 |
