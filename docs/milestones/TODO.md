# TODO — Planned Milestones

Back to [MILESTONES.md](../MILESTONES.md).

---

## P2 execution — Remote training dispatch — **Deferred**

> **P2 configuration shipped in v0.3 / P4 v1.0** — see [COMPLETED.md](COMPLETED.md).

**Goal:** Offload `train_model` to user-registered remote runtimes with quota-aware scheduling.

- `RuntimeProfiler` tracks free/paid usage limits locally
- `RuntimeScheduler` picks a runtime within quotas (`--remote-train`, `--runtime <id>`)
- Poll remote job status until artifacts sync back; `research resume` re-enters poll loop
- Init and post-train stages stay local in v1

```bash
research build --run-id <id> --remote-train
research run --competition <slug> --runtime kaggle-gpu-free
```

---

## P3 — v0.4 (Iteration Loop) — **Done**

See [COMPLETED.md](COMPLETED.md).

---

## P4 — v1.0 (Production Quality) — **Done**

See [COMPLETED.md](COMPLETED.md).

---

## Post-1.0 (Explicitly Deferred)

| Capability | Why deferred |
|------------|--------------|
| **Packaging & PyPI** | Bundle `templates/` + default configs in wheel; `pip install labpilot`; release workflow |
| **P2 remote execution** | Dispatch, polling, artifact sync (`--remote-train`) |
| Multi-agent systems | Orchestrator + templates are enough for P0–P4 |
| Vector databases | Brief uses competition page + profiler, not retrieval |
| Knowledge graphs | No cross-competition reasoning needed yet |
| Long-term memory | Each run is self-contained |
| Autonomous planning | Fixed pipeline DAG is sufficient |
| Self-modifying code | Templates + parameterization first |
| AutoML search | One strong baseline proves the loop |
| Multi-model orchestration | Single model per run |
| Full-pipeline remote execution | Training-only remote in P2 execution |
