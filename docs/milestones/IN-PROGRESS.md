# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

**Milestone 2 — Experiment Scientist** is in progress — Plans 1–3 shipped; Plans 4–8 still
design-only. **P4 — v1.0 (Production Quality)** shipped — see [COMPLETED.md](COMPLETED.md).

The goal: turn LabPilot from "execute one experiment" into "manage many experiments like a
research engineer" — an experiment graph with lineage, structured hypotheses, automatic
(deterministic) comparison, a reflection engine upgraded to use an LLM for analysis only, a
per-competition knowledge base, ranking of candidate next experiments, search, and a dashboard.
**No planner, no multi-agent system, no LLM code generation** — see the design doc for the
full set of guiding decisions.

The milestone is split into eight independently buildable plans, meant to land as separate PRs
in sequence:

| # | Plan | Depends on |
|---|------|------------|
| 1 | [Experiment Graph](milestone-2/plan-1-experiment-graph.md) | — |
| 2 | [Structured Hypothesis](milestone-2/plan-2-hypothesis.md) | 1 |
| 3 | [Automatic Comparator](milestone-2/plan-3-comparator.md) | 1 |
| 4 | [Reflection Engine upgrade](milestone-2/plan-4-reflection-engine.md) | 1, 2, 3 |
| 5 | [Knowledge Base](milestone-2/plan-5-knowledge-base.md) | 3 (4 optional) |
| 6 | [Experiment Ranking](milestone-2/plan-6-ranking.md) | 1, 2, 5 |
| 7 | [Experiment Search](milestone-2/plan-7-search.md) | 1 (3 optional) |
| 8 | [Dashboard / Report](milestone-2/plan-8-dashboard-report.md) | 1, 3, 5, 6 |

See [milestones/milestone-2/README.md](milestones/milestone-2/README.md) for the full
architecture writeup, repo-shape diff, and the milestone's closing deliverable.

Also queued, unrelated to Milestone 2: **P2 execution** (remote training dispatch) and
**Packaging & PyPI** — see [TODO.md](TODO.md) and [backlog.md](backlog.md).
