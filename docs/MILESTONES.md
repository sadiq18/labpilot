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
✔ Wrote HTML report
```

For a reviewable two-step alternative — pause after the brief to sanity-check the resolved
competition/dataset before spending time training — split it into `research init --competition
titanic` (parse → download → profile → brief) followed by `research build --run-id <id>` (baseline
→ code → train → evaluate → submission → upload → log → reflection).

Operator guide: [SOP.md](SOP.md) · Full CLI: [CLI.md](CLI.md).

---

## Status at a glance

| Track | Document | Summary |
|-------|----------|---------|
| **Completed** | [milestones/COMPLETED.md](milestones/COMPLETED.md) | P0 + P1 + P2 + P3 + P4 shipped |
| **In progress** | [milestones/IN-PROGRESS.md](milestones/IN-PROGRESS.md) | Research Engineer Phase B plans ready; Planner MVP shipped; RI Phase 1 shipped |
| **TODO** | [milestones/TODO.md](milestones/TODO.md) | P2 execution dispatch + post-1.0 items |
| **Backlog** | [milestones/backlog.md](milestones/backlog.md) | Unscheduled extensions (async kernel watcher, webhooks) |

---

## Milestone Roadmap

| Milestone | Version | Status | Goal |
|-----------|---------|--------|------|
| **P0** | v0.1 | **Done** | Prove the full pipeline for tabular classification/regression |
| **P1** | v0.2 | **Done** | Same loop, more competition types (text, image, metric-aware eval, rules) |
| **P2** | v0.3 | **Done** | Remote runtime registry, validation, and per-run runtime metadata |
| **P2 execution** | — | **Deferred** | Remote training dispatch (`--remote-train`, scheduler, artifact sync) |
| **P3** | v0.4 | **Done** | Iteration loop (`research improve`, tuning, run diffs) |
| **P4** | v1.0 | **Done** | Production quality (workspace, CI, dry-run, HTML reports) |
| **Experiment Scientist** | v0.5 | **Shipped** | Research memory, comparison, hypotheses, knowledge base, ranking, dashboard |
| **Research Intelligence** | v0.6 | **Phase 1 shipped** | Analyze landscape + hypotheses + brief; plans 1–11 + spike + F |
| **Research Planner** | — | **MVP shipped** | Hypothesis → planning compiler → executable DAG (`research plan`); Plans 1–6 |
| **Research Engineer** | — | **Phase B plans (not started)** | Approved plan → implemented, verified experiment (`research run --plan`); autonomous implement/verify/train/submit |

Details for each track live in the linked documents above:

- Experiment Scientist: [milestones/experiment-scientist/README.md](milestones/experiment-scientist/README.md)
- Research Intelligence: [milestones/research-intelligence/README.md](milestones/research-intelligence/README.md)
- Research Planner (MVP shipped): [milestones/research-planner/README.md](milestones/research-planner/README.md)
- Research Engineer (design): [milestones/research-engineer/README.md](milestones/research-engineer/README.md)

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for module layout, pipeline stages, and artifact contracts.
