# LabPilot Milestones

## North Star

**Close the research loop** — from competition landscape to durable knowledge —
without hand-written experiment code for the first 80% of the work.

```bash
research analyze <slug>
research plan create <slug> --baseline    # or --hypothesis H-xxx
research run --plan P-001 --competition <slug>
research journal --competition <slug>
```

After a successful plan-driven run (and reflection), a competition workspace should
have:

```
✔ Analyzed landscape / beliefs / hypotheses
✔ Compiled research plan (P-001 or hypothesis plan)
✔ Executed plan (workspace → code → verify → train → eval → submit)
✔ Logged experiment evidence
✔ Updated beliefs / hypotheses (Research Reflection)
✔ Research journal (evidence tiers + next experiment)
```

Operator guide: [SOP.md](SOP.md) · Full CLI: [CLI.md](CLI.md).

**Next product arc (Research OS):** The
[12-month Product and Startup Plan](../research-os/PRODUCT-PLAN.md) is the
canonical current direction. The Research OS README and milestone files are
historical implementation/design records.

---

## Status at a glance

| Track | Document | Summary |
|-------|----------|---------|
| **Research OS (implemented infrastructure)** | [../research-os/](../research-os/) | M1–M6 implemented; M22–M24 trust foundation is next |
| **Completed** | [milestones/COMPLETED.md](milestones/COMPLETED.md) | P0 + P1 + P2 + P3 + P4 shipped |
| **In progress** | [milestones/IN-PROGRESS.md](milestones/IN-PROGRESS.md) | Evidence Card/Graph; Engineer/Reflection/Planner shipped tracks |
| **TODO** | [milestones/TODO.md](milestones/TODO.md) | post-1.0 items |
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
| **Research Engineer** | — | **Phase B complete (dry-run SoR)** | Approved plan → implemented, verified experiment (`research run --plan`) |
| **Research Reflection** | — | **Phase B complete** | Outcomes → durable knowledge (`research reflect` / `journal`) |
| **Research OS** | — | **M1–M6 implemented; trust foundation active** | Conductor-led evolution — [product plan](../research-os/PRODUCT-PLAN.md) · [autonomy roadmap](../research-os/autonomy-roadmap/) |

Details for each track live in the linked documents above:

- Experiment Scientist: [milestones/experiment-scientist/README.md](milestones/experiment-scientist/README.md)
- Research Intelligence: [milestones/research-intelligence/README.md](milestones/research-intelligence/README.md)
- Research Planner (MVP shipped): [milestones/research-planner/README.md](milestones/research-planner/README.md)
- Research Engineer (design): [milestones/research-engineer/README.md](milestones/research-engineer/README.md)
- Research Reflection (design): [milestones/research-reflection/README.md](milestones/research-reflection/README.md)
- Research OS (design): [../research-os/README.md](../research-os/README.md) · [execution-plan](../research-os/execution-plan.md)

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the V1 kernel. Research OS layering:
[../research-os/architecture.md](../research-os/architecture.md).
