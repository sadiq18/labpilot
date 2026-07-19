# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

**Milestone 3 — Research Intelligence** — design Phase A locked; **implementation plans**
ready ([milestone-3/README.md](milestone-3/README.md) §15). Code not started.

**Design:** [milestone-3/README.md](milestone-3/README.md) ·
[milestone-3/knowledge-system.md](milestone-3/knowledge-system.md)

**Package layout (design):** `cli/` · `common/` · `research_engine/{execution,intelligence}/`
with Micro Agents in `intelligence/micro_agents/` and `execution/micro_agents/` (`*Agent` +
`skill.md`). On disk: `knowledge/<slug>/research/{raw,extracted,knowledge,experiments,reports}/`
+ `knowledge.db` — **local / gitignored**.

**Milestone 2 — Experiment Scientist** — Plans 1–8 shipped. See
[milestone-2/README.md](milestone-2/README.md).

**P4 — v1.0 (Production Quality)** shipped — see [COMPLETED.md](COMPLETED.md).

### Milestone 3 implementation plans

| # | Plan | Depends on |
|---|------|------------|
| 1 | [Foundation](milestone-3/plan-1-foundation.md) | M2 |
| 2 | [Knowledge Store](milestone-3/plan-2-knowledge-store.md) | 1 |
| 3 | [Micro Agents](milestone-3/plan-3-micro-agents.md) | 1 |
| 4 | [Experiment + Dataset](milestone-3/plan-4-experiment-dataset.md) | 1 |
| 5 | [Competition](milestone-3/plan-5-competition.md) | 1 |
| 6 | [Papers](milestone-3/plan-6-papers.md) | 1, 3 |
| 7 | [Repositories](milestone-3/plan-7-repositories.md) | 1, 3 |
| 8 | [Knowledge hub](milestone-3/plan-8-knowledge-hub.md) | 2, 3 (+ 4–7) |
| 9 | [Retrieval + Context Builder](milestone-3/plan-9-retrieval-context.md) | 8 |
| 10 | [Hypothesis Assistant](milestone-3/plan-10-hypothesis-assistant.md) | 9 |
| 11 | [Capstone](milestone-3/plan-11-capstone.md) | 10 |
| — | [Spike: Kaggle discussions](milestone-3/spike-kaggle-discussions.md) | — |
| F | [Forum Intelligence](milestone-3/plan-F-forum-intelligence.md) | Spike go or GitHub Issues + 1 |

Also queued, unrelated to Milestone 3: **P2 execution** and **Packaging & PyPI** — see
[TODO.md](TODO.md) and [backlog.md](backlog.md).
