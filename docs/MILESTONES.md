# LabPilot Milestones

## North Star

**One command closes the first 80% of a Kaggle competition loop** — from competition page to uploaded submission and a written reflection — without hand-written code.

```bash
research run --competition titanic
```

After a few hours, a complete run should produce:

```
✔ Parsed competition
✔ Downloaded data
✔ Built research brief
✔ Generated baseline
✔ Trained model
✔ Evaluated CV
✔ Generated submission
✔ Uploaded submission
✔ Logged experiment
✔ Wrote reflection
```

For a reviewable two-step alternative — pause after the brief to sanity-check the resolved
competition/dataset before spending time training — split it into `research init --competition
titanic` (parse → download → profile → brief) followed by `research build --run-id <id>` (baseline
→ code → train → evaluate → submission → upload → log → reflection).

---

## Status at a glance

| Track | Document | Summary |
|-------|----------|---------|
| **Completed** | [milestones/COMPLETED.md](milestones/COMPLETED.md) | P0 + P1 + P3 + P4 shipped |
| **In progress** | [milestones/IN-PROGRESS.md](milestones/IN-PROGRESS.md) | Nothing active |
| **TODO** | [milestones/TODO.md](milestones/TODO.md) | P2 execution + post-1.0 items |
| **Backlog** | [milestones/backlog.md](milestones/backlog.md) | Unscheduled extensions (async kernel watcher, webhooks) |

---

## Milestone Roadmap

| Milestone | Version | Status | Goal |
|-----------|---------|--------|------|
| **P0** | v0.1 | **Done** | Prove the full pipeline for tabular classification/regression |
| **P1** | v0.2 | **Done** | Same loop, more competition types (text, image, metric-aware eval, rules) |
| **P2** | v0.3 | **Deferred** | Remote runtime & scheduling (Kaggle Kernels, Colab, quota-aware dispatch) |
| **P3** | v0.4 | **Done** | Iteration loop (`research improve`, tuning, run diffs) |
| **P4** | v1.0 | **Done** | Production quality (workspace, CI, dry-run, runtime config) |

Details for each track live in the linked documents above.

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for module layout, pipeline stages, and artifact contracts.
