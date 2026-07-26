# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

## Research Planner — Phase B plans ready

**Status:** Design Phase A locked; **implementation plans 1–6 authored**. Application code
not started.

**Design:**

- [research-planner/README.md](research-planner/README.md) — product shape, compiler metaphor, CLI, non-goals
- [research-planner/schema.md](research-planner/schema.md) — `research_plans` / `research_tasks` / deps
- [research-planner/package-layout.md](research-planner/package-layout.md) — `planner/` + `accessor/`

### Implementation plans

| # | Plan | Depends on | Status |
|---|------|------------|--------|
| 1 | [Accessor layer](research-planner/plan-1-accessor.md) | RI shipped | Done |
| 2 | [Schemas + PlanStore](research-planner/plan-2-schemas-store.md) | 1 | Done |
| 3 | [Deterministic compiler](research-planner/plan-3-compiler.md) | 2 | Done |
| 4 | [Planning Engine Micro Agent](research-planner/plan-4-planning-engine.md) | 3 | Done |
| 5 | [CLI `research plan`](research-planner/plan-5-cli.md) | 3–4 | Not started |
| 6 | [Capstone + docs](research-planner/plan-6-capstone.md) | 5 | Not started |

**Next:** implement Plan 5 (CLI `research plan`), then Plan 6.

---

## Research Intelligence — Phase 1 shipped

**Milestone 3 — Research Intelligence** — Phase 1 (Plans 1–11) implemented;
[spike notes](research-intelligence/spike-kaggle-discussions-notes.md) + `research fetch` shipped.
Plan F (Forum Intelligence analyzer wiring) remains future.

**Design:** [research-intelligence/README.md](research-intelligence/README.md) ·
[research-intelligence/knowledge-system.md](research-intelligence/knowledge-system.md)

**Package layout (design):** `cli/` · `common/` · `research_engine/{execution,intelligence}/`
with Micro Agents in `intelligence/micro_agents/` and `execution/micro_agents/` (`*Agent` +
`skill.md`). On disk: `knowledge/<slug>/research/{raw,extracted,knowledge,experiments,reports}/`
+ `knowledge.db` — **local / gitignored**. Planner will add a sibling
`research_engine/planner/` pillar (see Research Planner design above).

### Milestone 3 implementation plans

| # | Plan | Depends on |
|---|------|------------|
| 1 | [Foundation](research-intelligence/plan-1-foundation.md) | M2 |
| 2 | [Knowledge Store](research-intelligence/plan-2-knowledge-store.md) | 1 |
| 3 | [Micro Agents](research-intelligence/plan-3-micro-agents.md) | 1 |
| 4 | [Experiment + Dataset](research-intelligence/plan-4-experiment-dataset.md) | 1 |
| 5 | [Competition](research-intelligence/plan-5-competition.md) | 1 |
| 6 | [Papers](research-intelligence/plan-6-papers.md) | 1, 3 |
| 7 | [Repositories](research-intelligence/plan-7-repositories.md) | 1, 3 |
| 8 | [Knowledge hub](research-intelligence/plan-8-knowledge-hub.md) | 2, 3 (+ 4–7) |
| 9 | [Retrieval + Context Builder](research-intelligence/plan-9-retrieval-context.md) | 8 |
| 10 | [Hypothesis Assistant](research-intelligence/plan-10-hypothesis-assistant.md) | 9 |
| 11 | [Capstone](research-intelligence/plan-11-capstone.md) | 10 |
| — | [Spike: Kaggle discussions](research-intelligence/spike-kaggle-discussions.md) | — |
| F | [Forum Intelligence](research-intelligence/plan-F-forum-intelligence.md) | Spike go or GitHub Issues + 1 |

---

## Experiment Scientist — shipped

**Milestone 2 — Experiment Scientist** — Plans 1–8 shipped. See
[experiment-scientist/README.md](experiment-scientist/README.md).

**P4 — v1.0 (Production Quality)** shipped — see [COMPLETED.md](COMPLETED.md).

Also queued, unrelated to Research Planner: **P2 execution** and **Packaging & PyPI** — see
[TODO.md](TODO.md) and [backlog.md](backlog.md).
