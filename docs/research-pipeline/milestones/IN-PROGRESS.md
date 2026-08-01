# In Progress

Back to [MILESTONES.md](../MILESTONES.md).

---

## Research OS — design

**Status:** Design on branch `research-os-design`.  
**Roadmap (architecture + tech co-evolve):** [../../research-os/README.md](../../research-os/README.md) ·
[execution-plan](../../research-os/execution-plan.md).

V1 pipeline remains the kernel. Next **implementation** is **M1 Platform Foundation**
(`research-os-m1-foundation`). One git branch per OS milestone thereafter.

---

## Evidence Card + Research Graph (causal learning)

**Status:** In progress on `research-execution`.
**Design:** [design/evidence-card.md](../design/evidence-card.md),
[design/research-graph.md](../design/research-graph.md).

Parent-vs-child COMPARE → Evidence Card → evidenced graph edges → belief/claim
steps. Graph DB and cross-competition shared knowledge remain on
[backlog.md](backlog.md).

---

## Research Reflection — Phase B complete

**Status:** Design Phase A **complete**; Plans **1–10 Done**.
**Milestone 6** — close the loop: evidence → critic → beliefs/hypotheses →
journal.

**Design:** [research-reflection/README.md](research-reflection/README.md) (+
architecture, schema, package-layout, beliefs-and-claims, cli).

### Implementation plans

| # | Plan | Depends on | Status |
|---|------|------------|--------|
| 1 | [Schema + stores](research-reflection/plan-1-schema-stores.md) | Design A | Done |
| 2 | [EvidenceExtractor](research-reflection/plan-2-evidence-extractor.md) | 1 | Done |
| 3 | [ExperimentCritic](research-reflection/plan-3-experiment-critic.md) | 2 | Done |
| 4 | [Belief + Hypothesis](research-reflection/plan-4-belief-hypothesis.md) | 1–3 | Done |
| 5 | [Engineer cutover](research-reflection/plan-5-engineer-cutover.md) | 2–4 | Done |
| 6 | [Lessons + synthesis](research-reflection/plan-6-lessons-synthesis.md) | 4–5 | Done |
| 7 | [Research Claims](research-reflection/plan-7-research-claims.md) | 1, 4, 6 | Done |
| 8 | [Journal CLI](research-reflection/plan-8-journal-cli.md) | 2–7 | Done |
| 9 | [Legacy cleanup](research-reflection/plan-9-legacy-cleanup.md) | 3–5, 8 | Done |
| 10 | [Capstone](research-reflection/plan-10-capstone.md) | 1–9 | Done |

**CLI:** `research reflect run` / `research journal` / `research claims` (+ Engineer Reporting hook).

---

## Research Engineer — Phase B complete (dry-run SoR)

**Status:** Plans **1–11 Done** on branch `research-execution`. Plan-driven
`research run --plan` is the SoR; legacy Pipeline removed. Deprecation tracker:
[research-engineer/pipeline-deprecation.md](research-engineer/pipeline-deprecation.md).
Capstone: [research-engineer/capstone-notes.md](research-engineer/capstone-notes.md).

**Product:** Autonomous Research Engineer — turn an approved Research Plan into a
verified experiment (implement, review, verify, train, evaluate, submit, evidence).
Not a thin “executor.”

**Design:** [research-engineer/README.md](research-engineer/README.md) (+ architecture,
schema, package-layout, baseline-plan, capabilities, runtime-and-recovery).

### Implementation plans

| # | Plan | Depends on | Status |
|---|------|------------|--------|
| 1 | [Schema / executions](research-engineer/plan-1-schema-executions.md) | Design A | Done |
| 2 | [Engineer controller](research-engineer/plan-2-engineer-controller.md) | 1 | Done |
| 3 | [Baseline P-001](research-engineer/plan-3-baseline-plan.md) | Planner MVP; ∥ after 1 | Done |
| 4 | [Workspace + Deps](research-engineer/plan-4-workspace-deps.md) | 2 | Done |
| 5 | [Code + Review](research-engineer/plan-5-code-review.md) | 4 | Done |
| 6 | [Verification / smoke](research-engineer/plan-6-verification.md) | 5 | Done |
| 7 | [Runtime](research-engineer/plan-7-runtime.md) | 6 | Done |
| 8 | [Train / Eval](research-engineer/plan-8-train-eval.md) | 6–7 | Done |
| 9 | [Submit / Report](research-engineer/plan-9-submit-report.md) | 8 | Done |
| 10 | [CLI cutover](research-engineer/plan-10-run-cli-cutover.md) | 2–9 | Done |
| 11 | [Capstone](research-engineer/plan-11-capstone.md) | 1–10 | Done |

**Next:** Research Reflection (above) owns post-run knowledge updates. Live
unattended train+upload and kernel-mode export remain Engineer follow-ons.

---

## Research Planner — plan-only MVP shipped

**Status:** Phase B Plans 1–6 **implemented**. Plan-only MVP complete
(`research plan create` / `show` / `list`). Capability implementation moves to
**Research Engineer** (above).

**Design + code:**

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
| 5 | [CLI `research plan`](research-planner/plan-5-cli.md) | 3–4 | Done |
| 6 | [Capstone + docs](research-planner/plan-6-capstone.md) | 5 | Done |

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
+ `knowledge.db` — **local / gitignored**. Planner is a sibling
`research_engine/planner/` pillar; Research Engineer fills `research_engine/execution/`.
Shared infra: `labpilot.accessor`.

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

Also queued: **Packaging & PyPI** — see [TODO.md](TODO.md) and [backlog.md](backlog.md).
Remote training dispatch is absorbed into Research Engineer Runtime capability design
(see [research-engineer/runtime-and-recovery.md](research-engineer/runtime-and-recovery.md)).
